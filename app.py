from flask import Flask, render_template, request
import cv2
from deepface import DeepFace
import time
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video_duration = 5  # in seconds

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    emotions = []

    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            face_img = frame[y:y+h, x:x+w]
            
            _, img_encoded = cv2.imencode('.jpg', face_img)
            img_bytes = img_encoded.tobytes()
            img_path = 'temp.jpg'
            with open(img_path, 'wb') as f:
                f.write(img_bytes)
            
            result = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)

            dominant_emotion = result[0]["dominant_emotion"][:]
            emotions.append(dominant_emotion)

            txt = "Emotion: " + dominant_emotion

            cv2.putText(frame, txt, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('frame', frame)

        if time.time() - start_time >= video_duration:
            break

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    most_common_emotion = max(set(emotions), key=emotions.count)

    movie_data = fetch_movies_from_imdb(most_common_emotion)

    return render_template('result.html', emotion=most_common_emotion, movies=movie_data)

def fetch_movies_from_imdb(emotion):
    genre_mapping = {
        'sad': 'drama',
        'disgust': 'musical',
        'angry': 'family',
        'neutral': 'thriller',
        'fear': 'sport',
        'happy': 'thriller',
        'surprised': 'film_noir'
    }

    genre = genre_mapping.get(emotion)

    if genre:
        url = f'https://www.rottentomatoes.com/browse/movies_in_theaters/genres:{genre}'
        print(f"Fetching movies for emotion: {emotion}, genre: {genre}, URL: {url}")
        response = requests.get(url)
        print(f"HTTP Response Code: {response.status_code}")
        soup = BeautifulSoup(response.text, 'html.parser')
        movies = soup.find_all('div', class_='article_movie_title')

        movie_data = []

        for i, movie in enumerate(movies[:10]):
            title = movie.a.text.strip()
            rating = movie.find_next('span', class_='tMeterScore').text.strip() if movie.find_next('span', class_='tMeterScore') else 'N/A'
            image_url = movie.find_next('img')['src'] if movie.find('img') else 'No Image'
            movie_data.append({'title': title, 'rating': rating, 'image_url': image_url})

        return movie_data

    return []

if __name__ == '__main__':
    app.run()
