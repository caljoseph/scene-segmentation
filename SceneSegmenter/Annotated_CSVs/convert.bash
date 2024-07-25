for file in *.xlsx; do
    xlsx2csv "$file" "${file%.xlsx}.csv"
done

