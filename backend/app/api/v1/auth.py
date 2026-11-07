from fastapi import APIRouter, status

from app.core.deps import AccessTokenBearerDep, OAuth2PasswordRequestFormDep, RefreshTokenBearerDep, SessionDep
from app.schemas.auth import LoginResponse, LogoutResponse, RefreshTokenResponse, SignupResponse
from app.schemas.users import UserCreate
from app.services import auth_service

auth_router = APIRouter()


@auth_router.post("/signup", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserCreate, session: SessionDep):
    return await auth_service.signup_user(user_data, session)


@auth_router.post("/login", response_model=LoginResponse)
async def login(form_data: OAuth2PasswordRequestFormDep, session: SessionDep):
    return await auth_service.login_user(form_data, session)


@auth_router.get("/logout", response_model=LogoutResponse)
async def logout(token_data: AccessTokenBearerDep):
    return {"message": "Logged out successfully"}


@auth_router.get("/refresh-token", response_model=RefreshTokenResponse)
async def refresh_token(token_data: RefreshTokenBearerDep):
    return await auth_service.refresh_access_token(token_data)
