python -c "import base64; s = open(\"$1\", \"rb\").read(); open(\"tmp.txt\", \"w\").write(base64.encodebytes(s).decode(\"UTF-8\").replace(\"\\n\", \"\"))"
