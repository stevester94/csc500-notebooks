for d in $(find . -maxdepth 1 -mindepth 1 -type d ); do
    (cd $d && echo Executing *papermill.py && ./*papermill.py)
done


