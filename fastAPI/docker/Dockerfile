FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the app.py and other required files into the container at /app
COPY ../app.py .
COPY ../requirements.txt .

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*


# Copy only the requirements.txt file first
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# ADD app.py

CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "15400"]