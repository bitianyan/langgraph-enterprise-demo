import sqlite3

c = sqlite3.connect("checkpoints.db")
tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
print("表：", tables)

n = c.execute("SELECT count(*) FROM checkpoints").fetchone()[0]
print("checkpoint 记录数：", n)
