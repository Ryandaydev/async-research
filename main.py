from datetime import date
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

import crud
import schemas
from database import AsyncSessionLocal

description = """
Fantasy football API helps you do awesome stuff. 🚀

## Players

You can **read players**.

## Scoring

You can look up a **player's scoring performance by week**.

## Membership

You can track **league and team composition over time**.

## Analytics

Look up **counts** of players, teams, and leagues.
"""

app = FastAPI(
    title="Fantasy Football API",
    description=description,
    summary="An API to query fantasy football data.",
    version="0.5.0",
    contact={
        "name": "Fantasy Football Support",
        "url": "https://x.com/fantasy_support",
        "email": "support@fantasy.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)


async def get_db():
    async with AsyncSessionLocal() as db:
        yield db


db_dependency = Annotated[AsyncSession, Depends(get_db)]


@app.get("/")
async def root():
    return {"message": "API health check successful"}


@app.get(
    "/v0/players/{player_id}/",
    response_model=schemas.Player,
    summary="Get player",
    response_description="Returns a player based on player_id",
    operation_id="get_player",
    tags=["Players"],
)
async def read_player(player_id: int, db: db_dependency):
    db_player = await crud.get_player(db=db, player_id=player_id)
    if db_player is None:
        raise HTTPException(status_code=404, detail="Player not found")
    return db_player


@app.get(
    "/v0/players/",
    response_model=list[schemas.Player],
    summary="Get all players",
    response_description="Returns all players and supports filtering by first_name, last_name, or minimum_last_changed_date",
    operation_id="get_all_players",
    tags=["Players"],
)
async def read_players(
    db: db_dependency,
    skip: int = 0,
    limit: int = 100,
    minimum_last_changed_date: date = None,
    last_name: str = None,
    first_name: str = None,
):
    players = await crud.get_players(
        db,
        skip=skip,
        limit=limit,
        min_last_changed_date=minimum_last_changed_date,
        last_name=last_name,
        first_name=first_name,
    )
    return players


@app.get(
    "/v0/performances/",
    response_model=list[schemas.Performance],
    summary="Get all player performances",
    response_description="Returns all performances and supports filtering by minimum_last_changed_date",
    operation_id="get_all_performances",
    tags=["Scoring"],
)
async def read_performances(
    db: db_dependency,
    skip: int = 0,
    limit: int = 100,
    minimum_last_changed_date: date = None,
):
    performances = await crud.get_performances(
        db,
        skip=skip,
        limit=limit,
        min_last_changed_date=minimum_last_changed_date,
    )
    return performances


@app.get(
    "/v0/leagues/{league_id}/",
    response_model=schemas.League,
    summary="Get league",
    response_description="Returns a league based on league_id",
    operation_id="get_league",
    tags=["Membership"],
)
async def read_league(league_id: int, db: db_dependency):
    db_league = await crud.get_league(db=db, league_id=league_id)
    if db_league is None:
        raise HTTPException(status_code=404, detail="League not found")
    return db_league


@app.get(
    "/v0/leagues/",
    response_model=list[schemas.League],
    summary="Get all leagues",
    response_description="Returns all leagues and supports filtering by league_name or minimum_last_changed_date",
    operation_id="get_all_leagues",
    tags=["Membership"],
)
async def read_leagues(
    db: db_dependency,
    skip: int = 0,
    limit: int = 100,
    minimum_last_changed_date: date = None,
    league_name: str = None,
):
    leagues = await crud.get_leagues(
        db,
        skip=skip,
        limit=limit,
        min_last_changed_date=minimum_last_changed_date,
        league_name=league_name,
    )
    return leagues


@app.get(
    "/v0/teams/",
    response_model=list[schemas.Team],
    summary="Get all teams",
    response_description="Returns all teams and supports filtering by team_name, league_id, or minimum_last_changed_date",
    operation_id="get_all_teams",
    tags=["Membership"],
)
async def read_teams(
    db: db_dependency,
    skip: int = 0,
    limit: int = 100,
    minimum_last_changed_date: date = None,
    team_name: str = None,
    league_id: int = None,
):
    teams = await crud.get_teams(
        db,
        skip=skip,
        limit=limit,
        min_last_changed_date=minimum_last_changed_date,
        team_name=team_name,
        league_id=league_id,
    )
    return teams


@app.get(
    "/v0/counts/",
    response_model=schemas.Counts,
    summary="Get player, team, and league counts",
    response_description="Returns counts for players, teams, and leagues",
    operation_id="get_counts",
    tags=["Analytics"],
)
async def read_counts(db: db_dependency):
    player_count = await crud.get_player_count(db)
    team_count = await crud.get_team_count(db)
    league_count = await crud.get_league_count(db)
    return {
        "player_count": player_count,
        "team_count": team_count,
        "league_count": league_count,
    }