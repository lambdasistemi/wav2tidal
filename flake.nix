{
  description = "wav2tidal — learn musical style from WAV corpora and drive TidalCycles live";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        py = pkgs.python3;

        # CPU dependencies — ingestion (R4), embeddings on CPU (R1), capture (R6).
        # Verified present in nixpkgs by the feature-001 research tasks.
        pythonEnv = py.withPackages (
          ps: with ps; [
            # ingestion / DSP
            soundfile
            librosa
            soxr
            resampy
            numpy
            scipy
            # embeddings (CLAP via transformers, run on CPU)
            transformers
            # live audio capture
            soundcard
            sounddevice
            # config / io
            pyyaml
            pyarrow
            # dev
            pytest
            ruff
            black
          ]
        );

        commonTools = [ pkgs.just ];
      in
      {
        # Default dev shell: CPU-only, everything CI needs. No ROCm/torch here,
        # so CI never pulls the heavy GPU stack (constitution IV).
        devShells.default = pkgs.mkShell {
          packages = [ pythonEnv ] ++ commonTools;
          shellHook = ''
            export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
          '';
        };

        # Opt-in training shell: torch-ROCm narrowed to gfx1151 (R2), plus the
        # LoRA/seq2seq + constrained-decode stack. NOT built by CI; hardware-
        # gated behind `just smoke-gpu` (FR-018). May be a long/first build.
        devShells.training = pkgs.mkShell {
          packages = [
            (py.withPackages (
              ps: with ps; [
                (torchWithRocm.override { gpuTargets = [ "gfx1151" ]; })
                transformers
                peft
                trl
                datasets
                soundfile
                librosa
                numpy
              ]
            ))
          ] ++ commonTools;
          shellHook = ''
            export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
            echo "wav2tidal training shell (gfx1151 ROCm). Run: just smoke-gpu"
          '';
        };

        packages.default = pythonEnv;

        # `nix flake check` builds this: the default shell's closure + a lint/test
        # runner over the pure core. CPU-only, no hardware.
        checks.ci = pkgs.runCommand "wav2tidal-ci"
          {
            nativeBuildInputs = [ pythonEnv ];
          }
          ''
            # ${self} is a read-only store path; copy to a writable tree so the
            # tools can write their caches, and redirect caches to $TMPDIR.
            cp -r ${self} src-tree
            chmod -R u+w src-tree
            cd src-tree
            export PYTHONPATH="$PWD/src"
            export PYTHONDONTWRITEBYTECODE=1
            export RUFF_CACHE_DIR="$TMPDIR/ruff"
            ruff check src tests
            black --check src tests
            pytest -p no:cacheprovider
            touch $out
          '';
      }
    );
}
