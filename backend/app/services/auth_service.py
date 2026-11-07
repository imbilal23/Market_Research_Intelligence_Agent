from fastapi import HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_jwt_token, verify_password
from app.schemas.auth import LoginResponse, RefreshTokenResponse, SignupResponse, TokenData
from app.schemas.users import UserCreate
from app.services import user_service


async def signup_user(user_data: UserCreate, session: AsyncSession) -> dict:
    new_user = await user_service.create_user(user_data, session)
    return {"message": "Account Created!", "user": new_user}


async def login_user(form_data: OAuth2PasswordRequestForm, session: AsyncSession) -> dict:
    email = form_data.username
    password = form_data.password

    user = await user_service.get_user_by_email(email, session)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Your account is not active")

    if not user.is_verified:
        raise HTTPException(status_code=403, detail="Your account is not verified")

    is_password_valid = verify_password(password, user.password_hash)
    if not is_password_valid:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_jwt_token(
        user_data={"email": user.email, "id": str(user.id)},
        refresh=False,
    )

    refresh_token = create_jwt_token(
        user_data={"email": user.email, "id": str(user.id)},
        refresh=True,
    )

    return {
        "message": "Login successful",
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": {"email": user.email, "id": str(user.id), "username": user.username},
    }


async def refresh_access_token(token_data: TokenData) -> dict:
    user = token_data.user
    user_data = {"email": user.email, "id": str(user.id)}
    new_access_token = create_jwt_token(user_data=user_data, refresh=False)
    return {"access_token": new_access_token}
