import subprocess
import glob

conlls = glob.glob("../auto/Penn_conbined_wsj/*/*.mrg")

for conll in conlls:
    dirpath = conll.split("/")[-2]
    file_id = conll.split("/")[-1].replace(".mrg", ".oracle")

    args = ["java -jar ParserOracleArcStdWithSwap.jar -t -1 -l 1",
            "-c " + conll,
            " > " + "../auto/Penn_Oracle/"+dirpath+"/"+file_id]
    args = " ".join(args)
    subprocess.run(args, shell=True)