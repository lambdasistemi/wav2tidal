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

        # Python package set where `torch` IS the gfx1151 ROCm build, so every
        # torch-dependent package (peft, trl, transformers) resolves to the SAME
        # torch — avoids a two-torch buildEnv collision on bin/torchrun.
        # Replicate nixpkgs' `torchWithRocm` overrides on the BASE torch (not via
        # torchWithRocm, which is `self.torch.override …` and would recurse) plus
        # the gfx1151 target. Same derivation as the prebuilt one -> cache hit.
        pyRocm = py.override {
          packageOverrides = final: prev: {
            torch = prev.torch.override {
              triton = final.triton-no-cuda;
              rocmSupport = true;
              cudaSupport = false;
              gpuTargets = [ "gfx1151" ];
            };
          };
        };

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
            # pattern grammar (subset parser/validator)
            lark
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
            (pyRocm.withPackages (
              ps: with ps; [
                torch
                transformers # ByT5 seq2seq (no trl/accelerate: manual loop)
                datasets
                soundfile
                librosa
                numpy
                lark # grammar membership for the eval validity metrics
                pyyaml # configs/train.yaml
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
            # librosa -> numba JITs and tries to cache next to its read-only
            # store source; give it (and HOME-based caches) a writable home.
            export HOME="$TMPDIR"
            export NUMBA_CACHE_DIR="$TMPDIR/numba"
            export MPLCONFIGDIR="$TMPDIR/mpl"
            ruff check src tests
            black --check src tests
            pytest -p no:cacheprovider
            touch $out
          '';
      }
    );
}
