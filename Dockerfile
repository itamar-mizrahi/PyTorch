# Use an official Python runtime as a parent image
# FROM python:3.11-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy the requirements file into the container at /app
# COPY requirements.txt .

# # Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application's code
# COPY . .

# # Expose the port the server runs on
# EXPOSE 8000

# # Define the command to run your application
# CMD ["python", "server.py"]


# 1. בחר תמונת בסיס של פייתון
FROM python:3.11-slim

# 2. הגדר את תיקיית העבודה בתוך הקונטיינר
WORKDIR /app

# 3. העתק את רשימת הדרישות והתקן אותן
# (העתקה בנפרד מנצלת את ה-cache של דוקר)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. העתק את שאר קבצי הפרויקט (השרת והמודל)
COPY . .

# 5. חשוף את הפורט שהשרת מאזין לו
EXPOSE 5000

# 6. הגדר את הפקודה שתרוץ כשהקונטיינר יתחיל
CMD ["python3", "server.py"]
