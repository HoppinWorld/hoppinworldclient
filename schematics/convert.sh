#!/bin/bash

converter="$1"
from="$2"

content=`cat "$from"`

while IFS="" read -r line || [ -n "$line" ]
do
  key=`echo "$line" |cut -d= -f1`
  value=`echo "$line" |cut -d= -f2`
  echo "key $key value $value"

  content=$(echo "$content"|sed s/$key/${content}/g)
done < $1

echo "$content"

