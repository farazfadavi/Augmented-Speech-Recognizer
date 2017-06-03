
import sys, os, xml.sax, re
from xml.dom.minidom import parse, parseString, getDOMImplementation

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
DESCRIPTION = 'Creates kaldi compatible lang directory'
FRAMESHIFT=0.005

# Add to path
sys.path = sys.path + [SCRIPT_DIR + '/../modules']

logopts = {'logging':{
    'nolog':"False",
    'logdir':".",
    'logname':'idlak_util',
    'loglevel':"Debug",
    'logtofile':"False",
    'logtostderr':"True"}
}

from alignsetup_def import saxhandler as idlak_saxhandler
from build_configuration import Logger

def binary_array(v, l):
    s = ""
    for i in range(l):
        if i == v - 1:
            s += " 1"
        else:
            s += " 0"
    return s.strip()

def forward_context(logger, input_fname, input_freqtable_fname, cexoutput_filename, rname = "alice"):

    # Load frequency table
    freqtables = eval(open(input_freqtable_fname).read())
    #print freqtables

    # Build mapping table
    lookuptables = {}
    lookuptables_len = {}
    for key in freqtables.keys():
        #print key
        vals = freqtables[key].keys()
        vals.sort()
        for v in vals:
            if not re.match('[0-9]+', v):
                # found a non integer value create a lookup table
                lookuptables[key] = {}
                mapping = 1
                for v in vals:
                    if v == '0':
                        lookuptables[key][v] = 0
                    else:
                        lookuptables[key][v] = mapping
                        mapping += 1
                lookuptables_len[key] = len(vals)
                if lookuptables[key].has_key('0'):
                    lookuptables_len[key] -= 1
                break
    
    # Read input file
    dom = parse(input_fname)
    # get header information
    header = dom.getElementsByTagName('txpheader')[0]
    cexheader = header.getElementsByTagName('cex')[0]
    # get by file ids
    fileids = dom.getElementsByTagName('fileid')
    if len(fileids) == 0:
        fileids = dom.getElementsByTagName('spt')
        for id, f in enumerate(fileids):
            idstr = rname + ('000' + str(id+1))[-3:]
            f.setAttribute('id', idstr)
    output_contexts = []
    for f in fileids:
        phons = f.getElementsByTagName('phon')
        output_contexts.append([f.getAttribute('id'), []])
        last_phon_name = ''
        for p in phons:
            phon_name = p.getAttribute('val')
            # Currently ignore utt internal split pauses
            if phon_name == 'pau' and last_phon_name == 'pau':
                last_phon_name = phon_name
                continue
            cex_string = p.firstChild.nodeValue
            cexs = cex_string.split()[1:]
            # get context phone name (may be different to xml phon val)
            pat = re.match('\^(.*?)\~(.*?)\-(.*?)\+(.*?)\=(.*)', cex_string.split()[0])
            if not pat:
                logger.log('critical', 'bad phone context string %s %s' %
                           (f, cex_string.split()[0]))
            phonename = pat.group(3)
            # currently add phone contexts as first 5 features
            # this to avoid a mismatch between manual phone
            # questions and the kaldi context information
            cexs = [pat.group(1), pat.group(2), pat.group(3), pat.group(4), pat.group(5)] + cexs
            cexs.insert(0, phonename)
            output_contexts[-1][-1].append(cexs)
            last_phon_name = phon_name

    print lookuptables, lookuptables_len
            
    # Perform mapping of input file using freqtable
    if cexoutput_filename == None or cexoutput_filename == '-':
        cexoutput_file = sys.stdout
    else:
        cexoutput_file = open(cexoutput_filename, 'w')

    #output_filename = os.path.join(outdir, 'output', 'cex.ark')
    fp = open(cexoutput_filename, 'w')
    for f in output_contexts:
        key = f[0]
        #print key, f
        fp.write(key + ' ')
        for p in f[1]:
            for i, v in enumerate(p):
                # replace symbols with integers^H binary arrays
                table = 'cex' + ('000' + str(i))[-3:]
                if lookuptables.has_key(table):
                    #v = str(lookuptables[table][v])
                    if not lookuptables[table].has_key(v):
                        logger.log('critical', ' no such key %s in row %s' % (v, table))
                        v = lookuptables[table].keys()[0]
                    v = binary_array(lookuptables[table][v], lookuptables_len[table])
                fp.write(v + ' ')
            fp.write('; ')
        fp.write('\n')
    fp.close()

def make_output_kaldidnn_cex(logger, input_filename, output_filename, cexoutput_filename, rname = "alice"):
    dom = parse(input_filename)
    # get header information
    header = dom.getElementsByTagName('txpheader')[0]
    cexheader = header.getElementsByTagName('cex')[0]
    
    # get by file ids
    fileids = dom.getElementsByTagName('fileid')
    if len(fileids) == 0:
        fileids = dom.getElementsByTagName('spt')
        for id, f in enumerate(fileids):
            idstr = rname + ('000' + str(id+1))[-3:]
            f.setAttribute('id', idstr)

    if output_filename == None or output_filename == '-':
        output_file = sys.stdout
    else:
        output_file = open(output_filename, 'w')

    # output the contexts only and build a string to integer table
    # this assumes space delimited contexts
    # note phone context is handled separately and the first
    # set of contexts (non space delimited) are phones
    # The current phone is prepended to the other contexts
    # to keep track of silences which may differ at start and end of
    # utterenaces from the alignment as it stands.
    # note data is also stored to be reformatted into kaldi in output_contexts
    freqtables = {}
    output_contexts = []
    for f in fileids:
        phons = f.getElementsByTagName('phon')
        output_contexts.append([f.getAttribute('id'), []])
        last_phon_name = ''
        for p in phons:
            phon_name = p.getAttribute('val')
            # Currently ignore utt internal split pauses
            if phon_name == 'pau' and last_phon_name == 'pau':
                last_phon_name = phon_name
                continue
            cex_string = p.firstChild.nodeValue
            cexs = cex_string.split()[1:]
            # get context phone name (may be different to xml phon val)
            pat = re.match('\^(.*?)\~(.*?)\-(.*?)\+(.*?)\=(.*)', cex_string.split()[0])
            if not pat:
                logger.log('critical', 'bad phone context string %s %s' %
                           (f, cex_string.split()[0]))
            phonename = pat.group(3)
            # currently add phone contexts as first 5 features
            # this to avoid a mismatch between manual phone
            # questions and the kaldi context information
            cexs = [pat.group(1), pat.group(2), pat.group(3), pat.group(4), pat.group(5)] + cexs
            # Currently set all contexts in pause to 0
            #if phonename == 'pau':
            #    for i in range(len(cexs)): cexs[i] = '0'
            # prepend the phone to keep track of silences and for sanity checks
            cexs.insert(0, phonename)
            # save/write contexts
            output_file.write('%s %s\n' % (f.getAttribute('id'), ' '.join(cexs)))
            output_contexts[-1][-1].append(cexs)
            # keep track of frequencies 
            for i in range(len(cexs)):
                key = 'cex' + ('000' + str(i))[-3:]
                if not freqtables.has_key(key):
                    freqtables[key] = {}
                if not freqtables[key].has_key(cexs[i]):
                    freqtables[key][cexs[i]] = 1
                else:
                    freqtables[key][cexs[i]] += 1
            last_phon_name = phon_name
            
    if output_filename != None and output_filename != '-':
        output_file.close()

    lookuptables = {}
    lookuptables_len = {}
    for i in range(len(cexs)):
        key = 'cex' + ('000' + str(i))[-3:]
        vals = freqtables[key].keys()
        vals.sort()
        for v in vals:
            if not re.match('[0-9]+', v):
                # found a non integer value create a lookup table
                lookuptables[key] = {}
                mapping = 1
                for v in vals:
                    if v == '0':
                        lookuptables[key][v] = 0
                    else:
                        lookuptables[key][v] = mapping
                        mapping += 1
                lookuptables_len[key] = len(vals)
                if lookuptables[key].has_key('0'):
                    lookuptables_len[key] -= 1
                break
    print lookuptables, lookuptables_len

    if cexoutput_filename == None or cexoutput_filename == '-':
        cexoutput_file = sys.stdout
    else:
        cexoutput_file = open(cexoutput_filename, 'w')
    #output_filename = os.path.join(outdir, 'output', 'cex.ark')
    fp = open(cexoutput_filename, 'w')
    for f in output_contexts:
        key = f[0]
        fp.write(key + ' ')
        for p in f[1]:
            for i, v in enumerate(p):
                # replace symbols with integers^H binary arrays
                table = 'cex' + ('000' + str(i))[-3:]
                if lookuptables.has_key(table):
                    #v = str(lookuptables[table][v])
                    v = binary_array(lookuptables[table][v], lookuptables_len[table])
                fp.write(v + ' ')
            fp.write('; ')
        fp.write('\n')
    fp.close()

    return cexs, output_contexts, freqtables, cexheader, lookuptables

def idlak_make_lang(textfile, datadir, langdir):
        p = xml.sax.make_parser()
        handler = idlak_saxhandler()
        p.setContentHandler(handler)
        p.parse(open(textfile, "r"))
        fp = open(os.path.join(datadir, "text"), 'w') 
        for i in range(len(handler.ids)):
            #if valid_ids.has_key(handler.ids[i]):
                # If we are forcing beginning and end silences add <SIL>s
            fp.write("%s %s\n" % (handler.ids[i], ' '.join(handler.data[i])))
            #fp.write("%s %s\n" % (handler.ids[i], ' '.join(handler.data[i])))
        fp.close()
        
        # lexicon and oov have all words for the corpus
        # whether selected or not by flist
        fpoov = open(os.path.join(langdir, "oov.txt"), 'w')
        fplex = open(os.path.join(langdir, "lexicon.txt"), 'w')
        # add oov word and phone (should never be required!
        fplex.write("<OOV> oov\n")
        # If we are forcing beginning and end silences make lexicon
        # entry for <SIL>
        fplex.write("<SIL> sil\n")
        fplex.write("<SIL> sp\n")
        # write transcription lexicon and oov lexicon for info
        words = handler.lex.keys()
        words.sort()
        phones = {}
        chars = {}
        for w in words:
            prons = handler.lex[w].keys()
            prons.sort()
            utf8w = w.decode('utf8')
            # get all the characters as a check on normalisation
            for c in utf8w:
                chars[c] = 1
            # get phone set from transcription lexicon
            for p in prons:
                pp = p.split()
                for phone in pp:
                    phones[phone] = 1
                fplex.write("%s %s\n" % (w, p))
            if handler.oov.has_key(w):
                fpoov.write("%s %s\n" % (w, prons[0]))
        fplex.close()
        fpoov.close()
        # write phone set
        # Should throw if phone set is not conformant
        # ie. includes sp or ^a-z@
        fp = open(os.path.join(langdir, "nonsilence_phones.txt"), 'w')
        phones = phones.keys()
        phones.sort()
        fp.write('\n'.join(phones) + '\n')
        fp.close()
        # write character set
        fp = open(os.path.join(langdir, "characters.txt"), 'w')
        chars = chars.keys()
        chars.sort()
        fp.write((' '.join(chars)).encode('utf8') + '\n')
        fp.close()
        # silence models
        fp = open(os.path.join(langdir, "silence_phones.txt"), 'w')
        fp.write("sil\nsp\noov\n")
        fp.close()
        # optional silence models
        fp = open(os.path.join(langdir, "optional_silence.txt"), 'w')
        fp.write("sp\n")
        fp.close()
        # an empty file for the kaldi utils/prepare_lang.sh script
        fp = open(os.path.join(langdir, "extra_questions.txt"), 'w')
        fp.close()

def load_labs(labfile):
    out = {}
    for l in open(labfile).readlines():
        ll = l.strip().split()
        key = ll[0]
        phones = []
        oldp = ll[1]
        np = 1
        start_time = 0.0
        for i, p in enumerate(ll[2:]):
            if p != oldp or i == len(ll) - 3:
                if p == oldp: np += 1
                end_time = round(start_time + np * FRAMESHIFT, 4)
                phones.append([start_time, end_time, oldp])
                start_time = end_time
                if p != oldp:
                    np = 1
                    oldp = p
                    # Border case where there is a single lonely phone at the end; should not happen
                    if i == len(ll) - 3:
                        end_time = round(start_time + np * FRAMESHIFT, 4)
                        phones.append([start_time, end_time, oldp])
            else:
                np += 1
        out[key] = phones
    return out

def load_words(wordfile):
    out = {}
    cur_times = {}
    for l in open(wordfile).readlines():
        ll = l.strip().split()
        key = ll[0]
        if not out.has_key(key):
            out[key] = []
            cur_times[key] = 0.0
        start_time = round(float(ll[2]), 4)
        end_time = round(start_time + float(ll[3]), 4)
        if start_time > cur_times[key]:
            out[key].append([cur_times[key], start_time, "<SIL>"])
        out[key].append((start_time, end_time, ll[4]))
        cur_times[key] = end_time
    # Hack: add a silence at the end of each sentence
    for k in out.keys():
        if out[k][-1][2] not in ['SIL', '!SIL', '<SIL>']:
            out[k].append([cur_times[k], 100000, "<SIL>"])
    return out

# Recreate an idlak compatible xml file from word and phone alignment
def write_xml_textalign(breaktype, breakdef, labfile, wordfile, output):
    impl = getDOMImplementation()

    document = impl.createDocument(None, "document", None)
    doc_element = document.documentElement
    
    #labs = glob.glob(labdir + '/*.lab')
    #labs.sort()
    all_labs = load_labs(labfile)
    all_words = load_words(wordfile)
    f = open(output, 'w')
    f.write('<document>\n')
    for id in sorted(all_labs.keys()):
        lab = all_labs[id]
        #print lab
        #stem = os.path.splitext(os.path.split(l)[1])[0]

        fileid_element = document.createElement("fileid")
        doc_element.appendChild(fileid_element)
        fileid_element.setAttribute('id', id)
        
        words = all_words[id]# open(os.path.join(wrddir, stem + '.wrd')).readlines()
        phones = all_labs[id]
        pidx = 0
        for widx, ww in enumerate(words):
            #ww = w.split()
            pron = []
            while pidx < len(phones):
                pp = phones[pidx]#.split()
                if pp[1] != ww[1] and float(pp[1]) > float(ww[1]):
                    break
                pron.append(pp[2].split('_')[0])
                pidx += 1
                #if pidx >= len(phones):
                #    break
            # Truncate end time to end time of last phone
            if float(ww[1]) > float(phones[-1][1]):
                ww[1] = float(phones[-1][1])
            #print ww, pron #, pidx, phones[pidx]
            if len(pron) == 0: continue
            if ww[2] not in ['SIL', '!SIL', '<SIL>']:
                lex_element = document.createElement("lex")
                fileid_element.appendChild(lex_element)
                lex_element.setAttribute('pron', ' '.join(pron))
                
                text_node = document.createTextNode(ww[2])
                lex_element.appendChild(text_node)
            else:
                if not widx or (widx == len(words) - 1):
                    break_element = document.createElement("break")
                    fileid_element.appendChild(break_element)
                    break_element.setAttribute('type', breakdef)
                else:
                    btype = breakdef
                    for b in breaktype.split(','):
                        bb = b.split(':')
                        minval = float(bb[1])
                        if float(ww[1]) - float(ww[0]) < minval:
                            btype = bb[0]
                    break_element = document.createElement("break")
                    fileid_element.appendChild(break_element)
                    break_element.setAttribute('type', btype)
        f.write(fileid_element.toxml() + '\n')

    f.write('</document>')
    f.close()

def main():
    from optparse import OptionParser
    usage="usage: %prog [options] text.xml datadir langdir\n" \
        "Takes the output from idlaktxp tool and create the corresponding\n " \
        "text file and lang directory required for kaldi forced alignment recipes."
    parser = OptionParser(usage=usage)
    parser.add_option('-m','--mode', default = 0,
                      help = 'Execution mode (0 => make_lang, 1 => write_xml_textalign, 2 => make_output_kaldidnn_cex')
    parser.add_option('-r','--root-name', default = "alice",
                      help = 'Root name to use for generating spurtID from anonymous spt')
    opts, args = parser.parse_args()
    if int(opts.mode) == 0 and len(args) == 3:
        idlak_make_lang(args[0], args[1], args[2])
    elif int(opts.mode) == 1 and len(args) == 5:
        write_xml_textalign(args[0], args[1], args[2], args[3], args[4])
    elif int(opts.mode) == 2:
        logger = Logger('kaldicex', logopts)
        if len(args) == 2:
            ret = make_output_kaldidnn_cex(logger, args[0], None, args[1], opts.root_name)
            if args[1] != '-' and args[1] != '':
                fname = args[1] + '.freq'
                fp = open(fname, 'w')
                fp.write(str(ret[2]))
                fp.close()
        # Forward with existing freqtable
        elif len(args) == 3:
            forward_context(logger, args[0], args[1], args[2], opts.root_name)
    else: 
        parser.error('Mandatory arguments missing or excessive number of arguments')

if __name__ == '__main__':
    main()
