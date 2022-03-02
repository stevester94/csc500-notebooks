dirs="cores metehan oracle.run1 oracle.run1.framed oracle.run2 oracle.run2.framed wisig"

for d in $dirs; do
    (cd $d && echo Executing *papermill.py)
done


