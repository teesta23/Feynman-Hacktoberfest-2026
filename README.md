# Feynman-Hacktoberfest-2026
Feynman Buddy is an AI study partner built on the Feynman technique: if you can teach it, you understand it. You explain a topic by voice or text to an AI student, which can be a curious classmate, a confused 10-year-old, a skeptical professor or an exam examiner. It asks probing doubts based on what you said. When you're stuck, it offers to explain the exact term you're missing, using your uploaded notes so it stays within your syllabus. Before each answer you rate your confidence, so it can flag when you're *confidently wrong*. After each session you get a knowledge map of what you've mastered and what to revisit, plus spaced-recall reminders. Built with: Streamlit, open-weight Llama 3.3 70B through Groq, and Whisper for voice. Snowflake stores each student's conversations, knowledge gaps and recall schedule, with Snowflake Cortex as a backup AI engine.

## Technologies Used
python, snowflake, groq, llama, whisper, streamlit

## Install Packages   
pip install -r requirements.txt  
winget install Gyan.FFmpeg

## Running  
Run "streamlit run app.py" to demo website!

