# ---- Stage 1: Build ----
FROM node:20-alpine AS builder
WORKDIR /app

# Install deps first (separate layer = cache-friendly on rebuilds)
COPY package*.json ./
RUN npm ci --quiet

# Copy source and build
COPY . .
RUN npm run build
# Output: /app/dist

# ---- Stage 2: Serve ----
FROM nginx:1.27-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
