import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


# 1、ratio()——使用纯Levenshtein Distance进行匹配。
#
# 2、partial_ratio()——基于最佳的子串（substrings）进行匹配
#
# 3、token_set_ratio——对字符串进行标记（tokenizes）并在匹配之前按字母顺序对它们进行排序
#
# 4、token_set_ratio——对字符串进行标记（tokenizes）并比较交集和余数

# [A-Z][a-z]*|
team_name = []
f = open("team_name.txt", "r")
for line in f:
    line = line[:-1]
    # country_name = re.search("(^[A-Z][a-z]*\t$)|[A-Z][a-z]*", line).group()
    team_name.append(line)
    # print(a)

a = "EERON"

result = process.extractOne(a, team_name, scorer=fuzz.token_set_ratio)
print(result)