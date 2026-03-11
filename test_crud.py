"""Testing SQLAlchemy Helper Functions"""

from datetime import date

import pytest
import pytest_asyncio

import crud
from database import AsyncSessionLocal

test_date = date(2024, 4, 1)


@pytest_asyncio.fixture
async def db_session():
    """Start an async database session and close it when done."""
    async with AsyncSessionLocal() as session:
        yield session


@pytest.mark.asyncio
async def test_get_player(db_session):
    player = await crud.get_player(db_session, player_id=1001)
    assert player.player_id == 1001


@pytest.mark.asyncio
async def test_get_players(db_session):
    players = await crud.get_players(
        db_session,
        skip=0,
        limit=10000,
        min_last_changed_date=test_date,
    )
    assert len(players) == 1018


@pytest.mark.asyncio
async def test_get_players_by_name(db_session):
    players = await crud.get_players(
        db_session,
        first_name="Bryce",
        last_name="Young",
    )
    assert len(players) == 1
    assert players[0].player_id == 2009


@pytest.mark.asyncio
async def test_get_all_performances(db_session):
    performances = await crud.get_performances(db_session, skip=0, limit=18000)
    assert len(performances) == 17306


@pytest.mark.asyncio
async def test_get_new_performances(db_session):
    performances = await crud.get_performances(
        db_session,
        skip=0,
        limit=10000,
        min_last_changed_date=test_date,
    )
    assert len(performances) == 2711


@pytest.mark.asyncio
async def test_get_league(db_session):
    league = await crud.get_league(db_session, league_id=5002)
    assert league.league_id == 5002
    assert len(league.teams) == 8


@pytest.mark.asyncio
async def test_get_leagues(db_session):
    leagues = await crud.get_leagues(
        db_session,
        skip=0,
        limit=10000,
        min_last_changed_date=test_date,
    )
    assert len(leagues) == 5


@pytest.mark.asyncio
async def test_get_teams(db_session):
    teams = await crud.get_teams(
        db_session,
        skip=0,
        limit=10000,
        min_last_changed_date=test_date,
    )
    assert len(teams) == 20


@pytest.mark.asyncio
async def test_get_teams_for_one_league(db_session):
    teams = await crud.get_teams(db_session, league_id=5001)
    assert len(teams) == 12
    assert teams[0].league_id == 5001


@pytest.mark.asyncio
async def test_get_team_players(db_session):
    first_team = (
        await crud.get_teams(
            db_session,
            skip=0,
            limit=1000,
            min_last_changed_date=test_date,
        )
    )[0]
    assert len(first_team.players) == 7


@pytest.mark.asyncio
async def test_get_player_count(db_session):
    player_count = await crud.get_player_count(db_session)
    assert player_count == 1018


@pytest.mark.asyncio
async def test_get_team_count(db_session):
    team_count = await crud.get_team_count(db_session)
    assert team_count == 20


@pytest.mark.asyncio
async def test_get_league_count(db_session):
    league_count = await crud.get_league_count(db_session)
    assert league_count == 5