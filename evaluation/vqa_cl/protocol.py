"""Published task order and data locations; no question-type heuristics."""

TASKS = ("recognition", "location", "judge", "commonsense", "count",
         "action", "color", "type", "subcategory", "causal")
ABBREVIATIONS = ("Rec", "Loc", "Jud", "Com", "Cou", "Act", "Col", "Typ", "Sub", "Cau")
ALIASES = dict(zip((name.lower() for name in ABBREVIATIONS), TASKS))
BENCHMARK = "cl4vqa_valen_v1"
REFERENCE_REVISION = "a9eff1c35a0cea13aa48e7233f5bbc8dd7b09471"
REFERENCE_URL = "https://github.com/ECNU-ICALK/CL-MoE"
DATA_URL = "https://drive.google.com/drive/folders/1mcAjzmCU1UVW0TKvsHAy9Sr1hmwsJudo"
# IDs observed in the authors' public CL4VQA directory. train_q_all is deliberately
# excluded: sequential training must never concatenate all future task data.
FILE_IDS = {
    "train": dict(zip(TASKS, (
        "1y5opV_MoOBXKl5aDD91u-QPbTx5W54Tv", "11Czp9uIjKQijETy-MrpEnwx5-pLS82FY",
        "18D6pc7Btj5rVCidIDtZOVJwfTYh2iDDY", "11cDl9BCS9iETDu1xPuOv7S_QezCymbcA",
        "1ln9KTN436T2g9rG-W3GnlZaq3cz5WOKG", "1GX95I1n0ReWUJ7G_iRnWJwMFUrX6B_jy",
        "16WBIOloufuGFsbljgKWIpH5qjtAV5ZdR", "1NVkH81paKkI1oC0Fyi4AtZfEFe_nNy7Y",
        "10EM-qlAd5VmihFqBVSscnEd6DAX-BCGw", "1xGQlk8otTO76yC6TpWzxHgDlHc6UE45P"))),
    "test": dict(zip(TASKS, (
        "1O1b76AUowjxuszoKFQ4fRuwbf4VC6gDK", "1wvDe8zWNXWxQQGKEg0tf6Svj54i39iEe",
        "1IEOQmq3BzeTWJ5-d2j9TFBqh1Mn83vwu", "1C4UAjdCmNoDrD6LrK7-3ySJoySHCSrA6",
        "1Z_gJwSjUbom63rP7xCKtfDlSZv1DlFS5", "1yV2yH9EtKeLL-r8heakMZNtbXWvOw00x",
        "1jE3FTyaKeHeGWTK2GXWX2oqN-SQnRU1G", "1K-qTuvyn0kx-QQqrrNvBH9_kJJlK9ylK",
        "1WuVpd208wEavcAEItDaOplYPN2_RM1w8", "1gxwLrirFb0-936pKmzlk43EUcSWM-9Gi"))),
}


def task_order(names):
    names = [ALIASES.get(name.lower().rstrip("."), name.lower()) for name in names]
    if not names or len(set(names)) != len(names) or set(names) - set(TASKS):
        raise ValueError(f"Tasks must be unique members of {TASKS}")
    return names
