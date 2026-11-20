FROM node:22-alpine AS build

WORKDIR /web

COPY package.json package-lock.json ./
COPY index.html tsconfig.json vite.config.ts ./
COPY src ./src

RUN npm ci && npm run build

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /web/dist /usr/share/nginx/html
