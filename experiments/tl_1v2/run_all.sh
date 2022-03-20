for d in $(find . -type f | grep papermill | xargs dirname ); do
    (cd $d && echo Executing *papermill.py && ./*papermill.py)
done


