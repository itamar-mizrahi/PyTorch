{
    "repository": {
        "repositoryArn": "arn:aws:ecr:us-east-1:822395332554:repository/my-ai-repo",
        "registryId": "822395332554",
        "repositoryName": "my-ai-repo",
        "repositoryUri": "822395332554.dkr.ecr.us-east-1.amazonaws.com/my-ai-repo",
        "createdAt": "2025-11-18T14:41:47.735000+02:00",
        "imageTagMutability": "MUTABLE",
        "imageScanningConfiguration": {
            "scanOnPush": true
        },
        "encryptionConfiguration": {
            "encryptionType": "AES256"
        }
    }
}
