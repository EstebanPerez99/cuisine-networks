from zlib import crc32

# Le et al. (2022), Table 1, doi:10.1016/j.ijhm.2021.103102.
TERMS = """amateurish ambitious artful artificial artisan assuming authentic bogus careful caring cheat classic conventional crafted craftsmanship creative deadset deceitful deceptive dinky-di ersatz ethical ethnic exotic expert extroverted fair-dinkum faithful faked false feigned forgery fusion genuine heartfelt historical hoax home-made home-style homey honest house-made humbug iconic idiosyncratic imitation impostor inspiring inspired integrity inventive legitimate master_chef misleading mistakable modern moral native offbeat old-fashioned original orthodox outlandish peculiar phony pretentious professional pure quack quirky quintessential real replicate ridgy-didge scam sham sincere skilful skilled specialty traditional true-blue truthful typical unique usual wholesome workmanship""".split()
assert len(set(TERMS)) == 88


def stable_hash(word):
    """Provide a process-independent initialization for Gensim."""
    return crc32(word.encode())
