# summary

bookcastai turns EPUB books into polished, podcast-style audio feeds. It parses an EPUB, cleans and chunks each chapter (optionally using an OpenRouter-backed AI pass for OCR fixes), runs TTS to render the narration, and emits an RSS feed so the finished book can be consumed like a podcast. Everything lives under `./data/{bookname}` so the pipeline is easy to rerun or share.

# work flow
1. generate metadata json as a result of parsing ebook
   python ./agent/epub_agent.py <book_title>
   add --ai-extract-text to clean each chunk with the OCR-to-TTS prompt when needed
2. generate audios
   python ./epub_to_pod.py <book_title> --voice=af_heart --speed=0.8 --overwrite 
3. generate and upload feed   
   python ./feed.py <book_title>
   
