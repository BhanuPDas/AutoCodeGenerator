# -------------------------------
# Stage 1: Build Vite React App
# -------------------------------
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .

RUN npm run build



# -------------------------------
# Stage 2: Serve with Nginx
# -------------------------------
FROM nginx:1.27-alpine
COPY --from=builder /app/dist /usr/share/nginx/html

# Nginx config for SPA routing
COPY <<EOF /etc/nginx/conf.d/default.conf
server {
    listen 80;
    server_name _;

    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files \$uri /index.html;
    }
}
EOF

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
