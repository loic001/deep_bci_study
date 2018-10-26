import sys
from datasets.rsvp import RSVP_COLOR_116
from datasets.meeg import P300_SPELLER_MEEG

raw = P300_SPELLER_MEEG().get_subject_data('S04', calib=False)
print(raw.info)
