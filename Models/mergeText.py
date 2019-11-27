import glob
def merge_text():
    read_files = glob.glob("*.txt")
    with open("result.txt", "wb") as outfile:
        for f in read_files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())
    f = open("result.txt")
    comments = open("comments.txt", "w")
    labels = open("labels.txt", "w")
    for line in f:
        a = line.strip().split("\t")
        comments.write(a[0] + '\n')
        labels.write(a[-1] + '\n')
