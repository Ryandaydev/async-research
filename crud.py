"""SQLAlchemy Query Functions"""

from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

import models


async def get_player(db: AsyncSession, player_id: int):
    result = await db.execute(
        select(models.Player)
        .options(selectinload(models.Player.performances))
        .where(models.Player.player_id == player_id)
    )
    return result.scalars().first()


async def get_players(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    min_last_changed_date: date = None,
    last_name: str = None,
    first_name: str = None,
):
    query = select(models.Player).options(selectinload(models.Player.performances))

    if min_last_changed_date:
        query = query.where(models.Player.last_changed_date >= min_last_changed_date)
    if first_name:
        query = query.where(models.Player.first_name == first_name)
    if last_name:
        query = query.where(models.Player.last_name == last_name)

    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    return result.scalars().all()


async def get_performances(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    min_last_changed_date: date = None,
):
    query = select(models.Performance)

    if min_last_changed_date:
        query = query.where(
            models.Performance.last_changed_date >= min_last_changed_date
        )

    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    return result.scalars().all()


async def get_league(db: AsyncSession, league_id: int = None):
    result = await db.execute(
        select(models.League)
        .options(joinedload(models.League.teams))
        .where(models.League.league_id == league_id)
    )
    return result.unique().scalars().first()


async def get_leagues(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    min_last_changed_date: date = None,
    league_name: str = None,
):
    query = select(models.League).options(joinedload(models.League.teams))

    if min_last_changed_date:
        query = query.where(models.League.last_changed_date >= min_last_changed_date)
    if league_name:
        query = query.where(models.League.league_name == league_name)

    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    return result.unique().scalars().all()


async def get_teams(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    min_last_changed_date: date = None,
    team_name: str = None,
    league_id: int = None,
):
    query = select(models.Team).options(selectinload(models.Team.players))

    if min_last_changed_date:
        query = query.where(models.Team.last_changed_date >= min_last_changed_date)
    if team_name:
        query = query.where(models.Team.team_name == team_name)
    if league_id:
        query = query.where(models.Team.league_id == league_id)

    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    return result.scalars().all()


# analytics queries
async def get_player_count(db: AsyncSession):
    result = await db.execute(select(func.count()).select_from(models.Player))
    return result.scalar_one()


async def get_team_count(db: AsyncSession):
    result = await db.execute(select(func.count()).select_from(models.Team))
    return result.scalar_one()


async def get_league_count(db: AsyncSession):
    result = await db.execute(select(func.count()).select_from(models.League))
    return result.scalar_one()