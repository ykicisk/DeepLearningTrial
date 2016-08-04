#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse as ap
import MeCab
import re
import unicodedata
import codecs
from collections import Counter
from collections import OrderedDict


old2now = {
    "いへば": "いえば",
    "言ふ": "言う",
    "言ひ": "言い",
    "言へ": "言え",
    "追ふ": "追う",
    "考へ": "考え",
    "舞ふ": "舞う",
}


def info(text, display_flag):
    if display_flag:
        print text

def tokenize_haiku(haiku, mecab, verbose=False):
    u"""mecabでtokenizeする"""
    acc_yomi = 0
    tokens = [u"<haiku_start>"]
    structure_ok_flag = True
    state = u"上句"
    for key, val in old2now.items():
        haiku = haiku.replace(key, val)
    haiku = haiku.decode("utf-8")
    haiku = re.sub(u"[『』〈〉*、]", u"", haiku)
    haiku = unicodedata.normalize('NFKC', haiku)
    haiku = haiku.encode("utf-8")
    token_info_list = mecab.parse(haiku).split("\n")
    for t in token_info_list:
        token_info = t.decode("utf-8").split(u"\t")
        if len(token_info) < 2:
            break
        word = token_info[0]
        yomi_kana = token_info[1]
        yomi_kana = re.sub(u"[ァィゥェォャュョ]", u"", yomi_kana)
        if not re.match(u"[ァ-ン]", yomi_kana):
            structure_ok_flag = False
            info(("MeCab can't read", word, "(yomi:", yomi_kana, ")"), verbose)
            break
        acc_yomi += len(yomi_kana)
        tokens.append(word)
        if state == u"上句":
            if acc_yomi > 6:
                structure_ok_flag = False
                info(("acc_yomi is ", acc_yomi, u" in 上句"), verbose)
                break
            if acc_yomi in [5, 6]:
                state = u"中句"
                acc_yomi = 0
                tokens.append(u"<1st_part_end>")
        elif state == u"中句":
            if acc_yomi > 8:
                structure_ok_flag = False
                info((u"acc_yomi is ", acc_yomi, u" in 中句"), verbose)
                break
            if acc_yomi in [7, 8]:
                state = u"下句"
                acc_yomi = 0
                tokens.append(u"<2nd_part_end>")
    if state != u"下句":
        info(u"下句 is not found", verbose)
        structure_ok_flag = False
    if structure_ok_flag and acc_yomi not in [5, 6]:
        info((u"acc_yomi is ", acc_yomi, u" in 下句"), verbose)
        structure_ok_flag = False
    tokens.append(u"<haiku_end>")
    if structure_ok_flag:
        return tokens
    else:
        if verbose:
            print "NG: ", haiku
            for t in tokens:
                print t
        return []


def main(src_path, dst_path, vocab_path, min_freq, mecab_dic_path, verbose):
    mecab = MeCab.Tagger("-Ochasen -d {}".format(mecab_dic_path))

    ng_count = 0
    token_cnt = Counter()
    haikus = []
    print "== parse haikus =="
    for idx, line in enumerate(open(src_path)):
        haiku = line.rstrip()
        tokens = tokenize_haiku(haiku, mecab, verbose)
        if tokens:
            # print "==", haiku, "=="
            haikus.append(tokens)
            for t in tokens:
                # print t
                token_cnt[t] += 1
        else:
            ng_count += 1
    print "(OK, NG) = (", len(haikus), ",", ng_count, ")"
    print "== calc vocab =="
    vocab = OrderedDict([
            ("<haiku_start>", 1),
            ("<1st_part_end>", 2),
            ("<2nd_part_end>", 3),
            ("<haiku_end>", 4),
            ("<unk>", 5)
            ])
    for idx, data in enumerate(token_cnt.most_common()):
        if idx < 4:  # special tokens
            continue
        key, val = data
        vocab[key] = idx + 2
        if val < min_freq:
            break
    print "== output vocab file =="
    with codecs.open(vocab_path, "w", "utf-8") as dst:
        for key in vocab.keys():
            dst.write(u"{}\n".format(key))
    print "== output tokenized haiku =="
    with open(dst_path, "w") as dst:
        for tokens in haikus:
            wordvecs = [vocab[t] if t in vocab else 5 for t in tokens]
            dst.write(u"{}\n".format("\t".join(map(str, wordvecs))))


if __name__ == "__main__":
    description = """tokenize haiku data.
    MeCabで俳句をtokenizeしてword vectorに変換(特殊種文字も追加)
    575に収まらないものは無視するが、1文字の字余りはゆるす。
    ※ 字足らずは面倒なので扱わない
    ※ よみではッ以外の小さい文字は文字数としてカウントしない削除
    例) len(ファッション) == len(フッシン) == 4

    # 特殊文字
    0: <haiku_start>
    1: <1st_part_end>
    2: <2nd_part_end>
    4: <haiku_end>
    5: <unk>
    # vocabrary
    6〜
    """
    parser = ap.ArgumentParser(description=description,
                               formatter_class=ap.RawDescriptionHelpFormatter)
    parser.add_argument('-s', '--src', required=True,
                        help='input file path')
    parser.add_argument('-d', '--dst', required=True,
                        help='output tokenized file path')
    parser.add_argument('-v', '--vocab', required=True,
                        help='vocabrary path')
    parser.add_argument('-m', '--min', default=2, type=int,
                        help='min freq for vocab (othres will be <unk>)')
    parser.add_argument('--verbose', default=False, action="store_true",
                        help='print debug text of haiku parse')
    neologd_path = "/usr/local/lib/mecab/dic/mecab-ipadic-neologd"
    parser.add_argument('--dictionary', default=neologd_path,
                        help='MeCab dictionary path')
    args = parser.parse_args()
    main(args.src, args.dst, args.vocab, args.min, args.dictionary,
         args.verbose)
