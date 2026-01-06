"""
Tests for API endpoints.
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    async def test_health_check(self, client: AsyncClient):
        response = await client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    async def test_legacy_health_check(self, client: AsyncClient):
        """Test legacy endpoint still works."""
        response = await client.get("/api/health")
        
        assert response.status_code == 200


@pytest.mark.asyncio
class TestPortfolioEndpoints:
    """Tests for portfolio endpoints."""
    
    async def test_get_portfolio_creates_default(self, client: AsyncClient):
        """First access should create a default portfolio."""
        response = await client.get("/api/v1/portfolio?lite=true")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "My Portfolio"
        assert data["total_value"] == 10000.0
        assert data["holdings"] == []
    
    async def test_update_portfolio_value(self, client: AsyncClient):
        """Test updating portfolio total value."""
        # First, create portfolio
        await client.get("/api/v1/portfolio?lite=true")
        
        # Update value
        response = await client.put(
            "/api/v1/portfolio",
            json={"total_value": 50000.0}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_value"] == 50000.0
    
    async def test_update_portfolio_name(self, client: AsyncClient):
        """Test updating portfolio name."""
        # First, create portfolio
        await client.get("/api/v1/portfolio?lite=true")
        
        # Update name
        response = await client.put(
            "/api/v1/portfolio",
            json={"name": "Retirement Fund"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Retirement Fund"


@pytest.mark.asyncio
class TestHoldingsEndpoints:
    """Tests for holdings endpoints."""
    
    async def test_delete_nonexistent_holding(self, client: AsyncClient):
        """Deleting a non-existent holding should return 404."""
        response = await client.delete("/api/v1/holdings/9999")
        
        assert response.status_code == 404
    
    async def test_update_nonexistent_holding(self, client: AsyncClient):
        """Updating a non-existent holding should return 404."""
        response = await client.put(
            "/api/v1/holdings/9999",
            json={"allocation_pct": 10.0}
        )
        
        assert response.status_code == 404


@pytest.mark.asyncio
class TestPortfolioHistory:
    """Tests for portfolio history endpoints."""
    
    async def test_get_empty_history(self, client: AsyncClient):
        """New portfolio should have empty history."""
        response = await client.get("/api/v1/portfolio/history")
        
        assert response.status_code == 200
        data = response.json()
        assert data["history"] == []

