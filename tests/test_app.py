from app import app


def test_app_created():
    assert app is not None


def test_home_page():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
