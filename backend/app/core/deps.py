from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import decode_token
from app.db.models import User
from app.db.session import get_session
from app.schemas.auth import TokenData


SessionDep = Annotated[AsyncSession, Depends(get_session)]


class TokenBearer(OAuth2PasswordBearer):
    def __init__(self, token_url=settings.token_bearer_url):
        super().__init__(tokenUrl=token_url)

    async def __call__(self, request: Request) -> TokenData:
        token = await super().__call__(request)
        if token is None:
            raise HTTPException(status_code=401, detail="Invalid Token")
        token_data = decode_token(token)
        if token_data is None:
            raise HTTPException(status_code=401, detail="Invalid Token")
        self.verify_token_type(token_data.refresh)

        return token_data

    @staticmethod
    def verify_token_type(is_refresh: bool) -> None:
        raise NotImplementedError("Please Override this method in child classes")


class AccessTokenBearer(TokenBearer):
    @staticmethod
    def verify_token_type(is_refresh: bool) -> None:
        if is_refresh:
            raise HTTPException(status_code=401, detail="Access token required")


AccessTokenBearerDep = Annotated[TokenData, Depends(AccessTokenBearer())]


class RefreshTokenBearer(TokenBearer):
    @staticmethod
    def verify_token_type(is_refresh: bool) -> None:
        if not is_refresh:
            raise HTTPException(status_code=401, detail="Refresh token required")


RefreshTokenBearerDep = Annotated[TokenData, Depends(RefreshTokenBearer())]


async def get_current_user(token_data: AccessTokenBearerDep, session: SessionDep) -> User:

    current_user = await session.get(User, token_data.user.id)
    if current_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if not current_user.is_active:
        raise HTTPException(status_code=403, detail="Your account is not active")
    if not current_user.is_verified:
        raise HTTPException(status_code=403, detail="Your account is not verified")

    return current_user


CurrentUserDep = Annotated[User, Depends(get_current_user)]
OAuth2PasswordRequestFormDep = Annotated[OAuth2PasswordRequestForm, Depends()]
