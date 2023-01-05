# build stage
FROM golang:alpine AS build-env
RUN apk --no-cache add build-base git mercurial gcc
RUN go get -u github.com/gin-gonic/gin

ADD ./src /src
RUN cd /src && go build -o main

# final stage
FROM alpine
WORKDIR /app
RUN mkdir -p /app/views
COPY --from=build-env /src/main /app/
COPY --from=build-env /src/views/* /app/views/

ENTRYPOINT ./main
