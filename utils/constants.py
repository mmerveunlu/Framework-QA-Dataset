# Filename template
NAMETMP = "{tag}.{name}.{tagID}"

# Language dictionary
LANG_DICT = {"tr": "turkish", "eng": "english"}
LANGUAGE_CHS = {"tr": 'çÇğĞıIİöÖşŞüÜ'}

# Replacements
chars_to_remove_regex = '[\}\{\`\_\/\$\#\~\(,\),\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'
CSV_EXT = ".csv"
JSON_EXT = ".json"
EMPTY = ""
FileSEP = "/"
MP3_EXT = ".mp3"
NEWLINE = "\n"
POINT = "."
SPACE = " "
special_character_dict = {"&": "ve",
                          "<": "küçüktür",
                          ">": "büyüktür",
                          "vs": "vesaire",
                          "+": "arti",
                          # "-": "eksi",
                          "=": "eşittir",
                          "_": "altçizgi"}

# Errors
ERROR_MESSAGE = "None-(Error: Bracket faulty[]\\{\\}| )"
WRONG_MESSAGE = "(\|.*=.*)"  # ex: '| adı = Hindistan'
WRONG_MESSAGE_2 = "(.*=.*)"
# Templates
KEYTMP = "wiki.dump.{date}.{tag}.{tagId}"
