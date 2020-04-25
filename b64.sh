python -c "import base64; s = open(\"$1\", \"rb\").read(); open(\"tmp.txt\", \"wb\").write(base64.encodebytes(s))"
