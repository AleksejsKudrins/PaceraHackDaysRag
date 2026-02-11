# Deployment Instructions

This application is designed to be deployed to a Linux server using Docker and GitHub Actions.

## Prerequisites

1.  **Linux Server**: A server with SSH access.
2.  **Docker & Docker Compose**: Installed on the server.
3.  **Sudo Access**: The user should have sudo access (configured without password as per setup).

## GitHub Secrets

To enable automatic deployment, you must add the following secrets to your GitHub repository (**Settings > Secrets and variables > Actions**):

- `SERVER_IP`: The IP address or hostname of your Linux server.
- `SERVER_USER`: The username used to SSH into the server.
- `SSH_PRIVATE_KEY`: The private SSH key used to authenticate with the server.

## Initial Server Setup (Optional)

If the deployment script fails to clone the repository because it's private, you may need to:
1.  Manually clone the repository into `/opt/rag-app` once.
2.  Ensure the `SERVER_USER` has ownership: `sudo chown -R $SERVER_USER:$SERVER_USER /opt/rag-app`.

## Deployment Process

The deployment is fully automated via GitHub Actions:
1.  Push code to the `main` branch.
2.  The "Build and Deploy" workflow will trigger.
3.  It will SSH into the server, pull the latest code, and run `docker-compose up --build -d`.

## Local Deployment

To run the app locally:

```bash
docker-compose up --build
```

The app will be available at `http://localhost:8501`.
