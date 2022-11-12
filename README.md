ERAS is an AI capable to recognize the emotions that a given audio file arouses, considering only its melody and not its semantic part.

Example of uses

from include import audioEmotionClassifiator as aec

ERAS = aec.ERAS()

ERAS = ERAS.train()

ERAS.predict(track="<AUDIO FILE PATH>")
