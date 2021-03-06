'''
# Copyright 2013-2014 Mirsk Digital Aps  (Author: Andreas Kirkedal)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.
'''


import codecs
import sys
import re
import writenumbers

## Global vars

normdict = {",": " ",
            ":": " ",
            ";": " ",
            "?": " ",
            "\\": " ",
            "\t": " ",
            #".": ""
            }

t_table = str.maketrans(normdict)


## Utility function

def getuttid_text(line):
    return line.split(" ", 1)


## Main

numtable = writenumbers.loadNumTable(sys.argv[1])
textin = codecs.open(sys.argv[2], "r", "utf8")
fid = codecs.open(sys.argv[3], "w", "utf8")
outtext = codecs.open(sys.argv[4], "w", "utf8")

for line in textin:
        utt_id, text = getuttid_text(line)
        normtext1 = text.translate(t_table)
        normtext2 = re.sub(r'  +', ' ', normtext1.strip())
        normtext3 = writenumbers.normNumber(normtext2, numtable)

        fid.write(utt_id + "\n")
        outtext.write(normtext3)

textin.close()
outtext.close()
fid.close()
