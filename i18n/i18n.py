import json
from pathlib import Path


def load_locale_file(lang):
    print(lang)
    lang_file = Path(f"./i18n/{lang}.json")
    if not lang_file.exists():
        default_lang_file = Path("./i18n/en.json")
        lang_dict = json.load(default_lang_file.open("r"))
    else:
        lang_dict = json.load(lang_file.open("r"))
    return lang_dict


class I18N:
    __lang_code_list = ["en", "zh-cn"]
    __lang_list = ["English", "简体中文"]

    def __init__(self, lang_symbol):
        self.__lang_code = "en"
        if lang_symbol in self.__lang_code_list:
            self.__lang_code = lang_symbol
        self.__lang = self.__lang_list[self.__lang_code_list.index(self.__lang_code)]
        self.__lang_dict = load_locale_file(self.__lang_code)

    def get_lang_list(self):
        return self.__lang_list

    def get_cur_lang(self):
        """
        Return app's current language

        :return: language display name
        :rtype: str
        """
        return self.__lang

    def get_lang_code_by_name(self, lang_name):
        return self.__lang_code_list[self.__lang_list.index(lang_name)]

    def get_lang_name_by_code(self, lang_code):
        return self.__lang_list[self.__lang_code_list.index(lang_code)]

    def get_str(self, lang_str):
        return self.__lang_dict.get(lang_str)

    def set_lang(self, lang_name):
        self.__lang_code = self.__lang_code_list[self.__lang_list.index(lang_name)]
        self.__lang = lang_name
        self.__lang_dict = load_locale_file(self.__lang_code)
