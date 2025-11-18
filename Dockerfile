FROM python:3.11-slim

# 1. העתקת המתאם של AWS (השורה שהייתה חסרה!)
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.1 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app

# 2. התקנת PyTorch בגרסת CPU (כדי לחסוך מקום וזמן)
RUN pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu

# 3. התקנת שאר הדרישות
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. העתקת הקוד
COPY . .

# 5. הגדרת הפורט עבור המתאם
ENV PORT=5000

# 6. הרצת השרת
CMD ["python3", "server.py"]
