# The big idea — a gentle tour

*Read this page first. No jargon, no diagrams — just what this thing is,
told slowly. The other pages go deep; this one goes easy.*

## The dream

Imagine you play a record you love — say **Take Five** — and a machine
listens and answers with its *own* version: synthesizers instead of a
saxophone, electronic textures instead of drums, but you'd still smile
and say *"hey, that's Take Five."* Something recognizable survives: the
feel, the harmony, the pulse.

That's the destination. wav2tidal today is the workshop where that
machine is being built, piece by piece. Some pieces are done and work
well; some don't exist yet. This page tells you which is which,
honestly.

## The instrument it plays

First, the machine needs something to make sound *with*. We gave it a
synthesizer rig called **SuperDirt** (the sound engine used by
TidalCycles live-coders): about thirty synthesizers — organs, saws,
kicks, noise machines — plus effects like filters, reverb and delay, and
the ability to play back slices of your own recordings.

Everything the machine ever plays is written down first as a short piece
of text — think of it as a **recipe**. A recipe might say:

> *take the "supersaw" synth, play a low note, slowly open its filter
> over six seconds, add a bit of reverb, and put a soft kick drum
> pattern underneath.*

In the project this recipe is written in a compact code (you'll see
things like `scene voice supersaw # note -12 mod cutoff ramp 200 2000`),
but that's all it is: a recipe. Recipes are the machine's *only* way to
make sound — which is great, because recipes can be checked ("is this
even playable?"), repaired, saved, and — crucially — *learned*.

## How the machine learns taste

How do you teach a machine which recipe produces which sound? The same
way you'd train a new cook:

1. **Cook thousands of random dishes.** The machine invents thousands of
   random (but always playable) recipes and renders each one to audio
   through its rig. Thanks to some recent speedups, it can "cook" a
   six-second dish in about half a second.
2. **Taste each one and write a note.** Every rendering gets a short
   tasting note, like: *"tempo 120, sparse, C minor, quite dark,
   brightness rising."* That last word matters — the notes describe not
   just how a sound *is* but how it *moves*.
3. **Train the apprentice.** A small neural network (it fits and trains
   on this machine's own GPU) reads all the (tasting note → recipe)
   pairs and learns to go *backwards*: give it a tasting note, and it
   writes a recipe that should taste like that.

That's the state of things today: you describe a sound — "slow, dark,
minor, getting brighter" — and the apprentice writes you a playable
recipe for it. About 8 out of 10 of its recipes are immediately playable
(a little automatic repair fixes most of the stumbles), and both numbers
are improving as we feed it more examples.

## How it will listen (the next chapter)

The part that connects this to your record player is the **live loop**,
which is designed but not yet built:

1. You play music at the machine — Take Five, your own noodling,
   anything.
2. The machine *tastes* it with the same tasting-note vocabulary, and
   also converts it into a kind of fingerprint (using a model called
   CLAP that has listened to millions of sounds and learned which ones
   humans would describe similarly).
3. The apprentice proposes a recipe. The rig plays it — out loud.
4. The machine listens *to itself*, fingerprints what it actually
   played, and compares: closer to the target, or further?
5. It nudges the recipe — a slower filter sweep, a different note, a
   touch more reverb — and tries again. And again. Evolution, with your
   target as the fitness test.

The recipes were deliberately designed so that "nudging" sounds musical:
a drone drifts, a filter sweeps a bit differently — no sudden teleports.

And here's the intended stage for all this: **a DJ at the input**. A
skilled DJ mixes records into the machine, so what comes in always has
great tempo, feeling, and drama — the machine never has to invent
structure, only follow it. The audience hears *only the machine*, a few
beats behind, like a shadow orchestra playing the DJ's drama in its own
voice. The delay isn't a problem to solve; it's the design.

## The honest gap

Here is the part you should know, because it's the difference between
what's built and what you described:

Today's "taste" system captures the **character** of sound — tempo,
darkness, key, motion, texture. If you play Take Five at it, the current
machine would answer with something that *feels* like it: the swing-ish
pulse, the minor colour, the smoky darkness. But it does **not** yet
follow the *tune* — the melody line, the chord changes as they unfold.
Recognizing "that exact melody" needs an extra listening skill (tracking
notes and harmony over time) that is not in the system yet.

That skill is buildable — the ingredients exist in the toolbox we
already use — but it's honest to say it's a chapter of its own, and it
comes after the live loop plays at all. When we design that chapter,
"would you recognize Take Five?" is the test we'll hold it to.

## Where each piece stands

| piece | plain words | status |
|---|---|---|
| Ingest | slice your records into playable material + first tasting notes | working |
| Recipes | the language of playable sound, with checking and repair | working |
| Rig | render any recipe to audio, fast and repeatably | working |
| Tasting notes | describe character *and* movement of a sound | working |
| Apprentice | tasting note → recipe, trained on this machine | working, improving |
| Live loop | listen, propose, play, compare, nudge | designed, next |
| Following the tune | melody & chord tracking for recognizability | future chapter |

If you want the deep version of any piece, the dense pages are still
there: [Architecture](architecture.md), [Sound engine](sound-engine.md),
[Language](language.md), [Model](model.md), [Roadmap](roadmap.md).
