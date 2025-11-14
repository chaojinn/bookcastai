# summary

bookcastai turns EPUB books into polished, podcast-style audio feeds. It parses an EPUB, cleans and chunks each chapter (optionally using an OpenRouter-backed AI pass for OCR fixes), runs TTS to render the narration, and emits an RSS feed so the finished book can be consumed like a podcast. Everything lives under `./data/{bookname}` so the pipeline is easy to rerun or share.

# work flow
1. generate metadata json as a result of parsing ebook
   python ./agent/epub_agent.py xxx.epub -o=./data/{bookname}/epub.json
   add --ai-extract-text to clean each chunk with the OCR-to-TTS prompt when needed
2. generate audios
   python ./epub_to_pod.py ./data/{bookname}/book.json ./data/{bookname} --voice=af_heart --speed=0.8 --overwrite 
3. generate and upload feed   
   python ./feed.py ./data/{bookname}/book.json ./data/{bookname}/book.xml --audio-dir="./data/{bookname}"
   
