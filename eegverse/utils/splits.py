from typing import List, Dict
def loso_partitions(subjects: List[str]) -> Dict[str, Dict[str, List[str]]]:
    return {s: {"train":[x for x in subjects if x!=s], "test":[s]} for s in subjects}
