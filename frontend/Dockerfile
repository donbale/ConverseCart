# frontend/Dockerfile
FROM nginx:alpine

# Copy your static files (index.html, App.js, assets…) into Nginx’s web root
COPY . /usr/share/nginx/html

# Expose port 80 inside the container (compose will map it to 8080)
EXPOSE 80

# Nginx’s default command serves the files