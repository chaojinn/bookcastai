# work flow
1. generate metadata json as a result of parsing ebook
   python ./agent/epub_agent.py xxx.epub -o=./data/{bookname}/epub.json
   # add --ai-extract-text to clean each chunk with the OCR-to-TTS prompt when needed
2. generate audios
   python ./epub_to_pod.py ./data/{bookname}/book.json ./data/{bookname} --voice=af_heart --speed=0.8 --overwrite 
3. generate and upload feed   
   python ./feed.py ./data/{bookname}/book.json ./data/{bookname}/book.xml --audio-dir="./data/{bookname}"
   
