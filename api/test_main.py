import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from inference_api.main_inference_api import app


@pytest_asyncio.fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_read_main(async_client):
    response = await async_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API health check successful"}


@pytest.mark.asyncio
async def test_read_players(async_client):
    response = await async_client.get("/v0/players/?skip=0&limit=10000")
    assert response.status_code == 200
    assert len(response.json()) == 1018


@pytest.mark.asyncio
async def test_read_players_by_name(async_client):
    response = await async_client.get(
        "/v0/players/?first_name=Bryce&last_name=Young"
    )
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0].get("player_id") == 2009


@pytest.mark.asyncio
async def test_read_players_with_id(async_client):
    response = await async_client.get("/v0/players/1001/")
    assert response.status_code == 200
    assert response.json().get("player_id") == 1001


@pytest.mark.asyncio
async def test_read_performances(async_client):
    response = await async_client.get("/v0/performances/?skip=0&limit=20000")
    assert response.status_code == 200
    assert len(response.json()) == 17306


@pytest.mark.asyncio
async def test_read_performances_by_date(async_client):
    response = await async_client.get(
        "/v0/performances/?skip=0&limit=20000&minimum_last_changed_date=2024-04-01"
    )
    assert response.status_code == 200
    assert len(response.json()) == 2711


@pytest.mark.asyncio
async def test_read_leagues_with_id(async_client):
    response = await async_client.get("/v0/leagues/5002/")
    assert response.status_code == 200
    assert len(response.json()["teams"]) == 8


@pytest.mark.asyncio
async def test_read_leagues(async_client):
    response = await async_client.get("/v0/leagues/?skip=0&limit=500")
    assert response.status_code == 200
    assert len(response.json()) == 5


@pytest.mark.asyncio
async def test_read_teams(async_client):
    response = await async_client.get("/v0/teams/?skip=0&limit=500")
    assert response.status_code == 200
    assert len(response.json()) == 20


@pytest.mark.asyncio
async def test_read_teams_for_one_league(async_client):
    response = await async_client.get("/v0/teams/?skip=0&limit=500&league_id=5001")
    assert response.status_code == 200
    assert len(response.json()) == 12


@pytest.mark.asyncio
async def test_counts(async_client):
    response = await async_client.get("/v0/counts/")
    response_data = response.json()
    assert response.status_code == 200
    assert response_data["league_count"] == 5
    assert response_data["team_count"] == 20
    assert response_data["player_count"] == 1018