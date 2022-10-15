import os
import random

import torch
from text import transform
from text.cleaners import japanese_tokenization_cleaners, japanese_cleaners2, japanese_cleaners
from models import SynthesizerTrn
import commons
import utils
import soundfile as sf
import janome

characterDict = {
    'saki': ['./configs/saki.json', './model/saki_v1.0.pth', 1],
    'ichika': ['./configs/ichika.json', './model/ichika_v1.0.pth', 2],
    'honami': ['./configs/honami.json', './model/honami_v1.0.pth', 3],
    'shiho': ['./configs/shiho.json', './model/shiho_v1.0.pth', 3],

    'kanade': ['./configs/kanade.json', './model/kanade_v1.0.pth', 4],
    'ena': ['./configs/ena.json', './model/ena_v1.0.pth', 3],
    'mafuyu0': ['./configs/mafuyu.json', './model/mafuyu_v1.0.pth', 3],
    'mafuyu1': ['./configs/mafuyu.json', './model/mafuyu_v1.0.pth', 3],
    'mizuki': ['./configs/mizuki.json', './model/mizuki_v1.0.pth', 1],

    'airi': ['./configs/airi.json', './model/airi_v1.0.pth', 1],
    'minori': ['./configs/mmj.json', './model/mmj_v1.0.pth', 3],
    'haruka': ['./configs/mmj.json', './model/mmj_v1.0.pth', 3],
    'shizuku': ['./configs/mmj.json', './model/mmj_v1.0.pth', 3],

    'akito': ['./configs/vbs.json', './model/vbs_v1.1.pth', 3],
    'an': ['./configs/vbs.json', './model/vbs_v1.1.pth', 3],
    'kohane': ['./configs/vbs.json', './model/vbs_v1.1.pth', 3],
    'toya': ['./configs/vbs.json', './model/vbs_v1.1.pth', 3],

    'emu': ['./configs/ws.json', './model/ws_v1.1.pth', 3],
    'nene': ['./configs/ws.json', './model/ws_v1.1.pth', 3],
    'rui': ['./configs/ws.json', './model/ws_v1.1.pth', 3],
    'tsukasa': ['./configs/ws.json', './model/ws_v1.1.pth', 3],
}

multiDict = ['mafuyu0', 'mafuyu1', 'minori', 'haruka', 'shizuku', 'akito', 'an', 'kohane', 'toya', 'emu', 'nene', 'rui',
             'tsukasa']


class Generator:

    def __init__(self):
        self.symbols = list(' !"&*,-.?ABCINU[]abcdefghijklmnoprstuwyz{}~')
        self.character = None
        self.type = 1
        self.multiSpeakers = False
        self.multiId = 0

    def getSymbols(self):
        return self.symbols

    def getSpeakers(self):
        return self.multiSpeakers

    def getMultiId(self):
        return self.multiId

    def changeSpeakers(self, flag, cStr='none'):
        if flag:
            self.multiSpeakers = True
            if cStr == 'minori' or cStr == 'mafuyu0' or cStr == 'akito' or cStr == 'emu':
                self.multiId = 0
            elif cStr == 'haruka' or cStr == 'mafuyu1' or cStr == 'an' or cStr == 'nene':
                self.multiId = 1
            elif cStr == 'airi' or cStr == 'kohane' or cStr == 'rui':
                self.multiId = 2
            elif cStr == 'shizuku' or cStr == 'toya' or cStr == 'tsukasa':
                self.multiId = 3
        elif not flag:
            self.multiSpeakers = False
            self.multiId = 0

    def getCharacter(self):
        return self.character

    def changeCharater(self, chara):
        self.character = chara

    def changeSymbols(self, type):
        if type == 1:
            self.symbols = list(' !"&*,-.?ABCINU[]abcdefghijklmnoprstuwyz{}~')
            self.type = 1
        elif type == 2:
            _pad = '_'
            _punctuation = ',.!?-'
            _letters = 'AEINOQUabdefghijkmnoprstuvwyzʃʧ↓↑ '
            self.symbols = [_pad] + list(_punctuation) + list(_letters)
            self.type = 2
        elif type == 3:
            _pad = '_'
            _punctuation = ',.!?-~…'
            _letters = 'AEINOQUabdefghijkmnoprstuvwyzʃʧʦ↓↑ '
            self.symbols = [_pad] + list(_punctuation) + list(_letters)
            self.type = 3
        elif type == 4:
            _pad = '_'
            _punctuation = ';:,.!?¡¿—…"«»“” '
            _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
            self.symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
            self.type = 4


def get_text(text, hps, symbols):
    text_norm = transform.cleaned_text_to_sequence(text, symbols)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def jtts(text, type, symbols, multiSpeaker):
    stn_tst = None
    try:
        if type == 1 or type == 4:
            stn_tst = get_text(japanese_tokenization_cleaners(text), hps, symbols)
        elif type == 2:
            stn_tst = get_text(japanese_cleaners(text), hps, symbols)
        elif type == 3:
            stn_tst = get_text(japanese_cleaners2(text), hps, symbols)
        if not os.path.exists('playSounds'):
            os.makedirs('playSounds')
    except KeyError as e:
        return 'KeyError: ' + str(e)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        if multiSpeaker[0]:
            sid = torch.LongTensor([multiSpeaker[1]])
            try:
                audio = \
                    net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][
                        0, 0].data.float().numpy()
            except Exception as e:
                return e
        else:
            try:
                audio = \
                    net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[
                        0][
                        0, 0].data.float().numpy()
            except Exception as e:
                return e

        #######
        # 这里的radio是一个浮点数列表，相当于音频的内容，也可以不保存直接输出?
        #######
        filename = 'playSounds/' + text.replace('?', '') + str(random.randint(1, 100)) + '.wav'
        sf.write(filename, audio, 22050)
        ###### 看选择返回什么了，文件还是audio列表
        return audio


def output_tts(character, text, generator):
    if character in characterDict and character != generator.getCharacter():
        # 选择单目标或多目标
        if character in multiDict:
            generator.changeSpeakers(True, character)
        else:
            generator.changeSpeakers(False)
        generator.changeCharater(character)
        generator.changeSymbols(characterDict[character][2])
        # 加载参数
        print(characterDict[character][0])
        global hps
        hps = utils.get_hparams_from_file(characterDict[character][0])
        global net_g
        net_g = SynthesizerTrn(
            len(generator.getSymbols()),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model)
        net_g.eval()
        # 加载模型
        print(characterDict[character][1])
        _ = utils.load_checkpoint(characterDict[character][1], net_g, None)
        # text ,type(int) ,symbol(list),multi(list[bool,int])
        return jtts(text, characterDict[character][2], generator.getSymbols(),
                    [generator.getSpeakers(), generator.getMultiId()])
    elif character in characterDict:
        return jtts(text, characterDict[character][2], generator.getSymbols(),
                    [generator.getSpeakers(), generator.getMultiId()])
    else:
        return 'character error'


if __name__ == '__main__':
    generator = Generator()
    ## 下面两个是全局变量
    hps = None
    net_g = None
    while True:
        # 模拟输入角色
        character = input()
        # 模拟输入台词
        text = '今日の天気は良いです'
        audio_list = output_tts(character, text, generator)
        print(audio_list)
