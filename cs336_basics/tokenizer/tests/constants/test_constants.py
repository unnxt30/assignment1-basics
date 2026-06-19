from cs336_basics.tokenizer.constants import PAT
import regex as re

def test_pretokenize():
    test_str = "some text that i'll pre-toknize"
    
    pt = re.findall(PAT, test_str)
    assert len(pt) > 0
    # assert "pre-tokenize" not in pt