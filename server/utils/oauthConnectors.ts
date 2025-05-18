import { Request, Response } from 'express';
import { AuthRequest } from '../middleware/auth';
import { storage } from '../storage';
import fetch from 'node-fetch';
import { stringify } from 'querystring';

// OAuth configuration
const oauthConfig = {
  googleAds: {
    clientId: process.env.GOOGLE_ADS_CLIENT_ID || 'your-client-id',
    clientSecret: process.env.GOOGLE_ADS_CLIENT_SECRET || 'your-client-secret',
    redirectUri: 'https://your-app-url.com/api/oauth/google-ads/callback',
    authUrl: 'https://accounts.google.com/o/oauth2/auth',
    tokenUrl: 'https://oauth2.googleapis.com/token',
    scope: 'https://www.googleapis.com/auth/adwords',
  },
  facebookAds: {
    clientId: process.env.FACEBOOK_ADS_CLIENT_ID || 'your-client-id',
    clientSecret: process.env.FACEBOOK_ADS_CLIENT_SECRET || 'your-client-secret',
    redirectUri: 'https://your-app-url.com/api/oauth/facebook-ads/callback',
    authUrl: 'https://www.facebook.com/v15.0/dialog/oauth',
    tokenUrl: 'https://graph.facebook.com/v15.0/oauth/access_token',
    scope: 'ads_read',
  },
  googleAnalytics: {
    clientId: process.env.GOOGLE_ANALYTICS_CLIENT_ID || 'your-client-id',
    clientSecret: process.env.GOOGLE_ANALYTICS_CLIENT_SECRET || 'your-client-secret',
    redirectUri: 'https://your-app-url.com/api/oauth/google-analytics/callback',
    authUrl: 'https://accounts.google.com/o/oauth2/auth',
    tokenUrl: 'https://oauth2.googleapis.com/token',
    scope: 'https://www.googleapis.com/auth/analytics.readonly',
  },
};

// Generate OAuth authorization URL
export const getAuthorizationUrl = (provider: 'googleAds' | 'facebookAds' | 'googleAnalytics', state: string) => {
  const config = oauthConfig[provider];
  
  const params = {
    client_id: config.clientId,
    redirect_uri: config.redirectUri,
    scope: config.scope,
    response_type: 'code',
    state,
  };
  
  return `${config.authUrl}?${stringify(params)}`;
};

// Exchange authorization code for tokens
export const exchangeCodeForTokens = async (
  provider: 'googleAds' | 'facebookAds' | 'googleAnalytics',
  code: string
) => {
  const config = oauthConfig[provider];
  
  const params = {
    client_id: config.clientId,
    client_secret: config.clientSecret,
    code,
    redirect_uri: config.redirectUri,
    grant_type: 'authorization_code',
  };
  
  try {
    const response = await fetch(config.tokenUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: stringify(params),
    });
    
    if (!response.ok) {
      const error = await response.text();
      throw new Error(`OAuth token exchange failed: ${error}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error(`Error exchanging code for ${provider} tokens:`, error);
    throw error;
  }
};

// Initialize OAuth flow
export const initializeOAuth = (req: AuthRequest, res: Response) => {
  const { provider, projectId } = req.params;
  
  if (!['googleAds', 'facebookAds', 'googleAnalytics'].includes(provider)) {
    return res.status(400).json({ message: 'Invalid provider' });
  }
  
  if (!projectId || isNaN(parseInt(projectId))) {
    return res.status(400).json({ message: 'Invalid project ID' });
  }
  
  try {
    // Create state with project ID and CSRF token
    const state = JSON.stringify({
      projectId,
      csrf: req.session.csrfToken || Math.random().toString(36).substring(2, 15),
    });
    
    // Store state in session
    req.session.oauthState = state;
    
    // Redirect to provider's authorization page
    const authUrl = getAuthorizationUrl(
      provider as 'googleAds' | 'facebookAds' | 'googleAnalytics',
      Buffer.from(state).toString('base64')
    );
    
    return res.redirect(authUrl);
  } catch (error) {
    console.error(`Error initializing ${provider} OAuth:`, error);
    return res.status(500).json({ message: 'Error initializing OAuth' });
  }
};

// Handle OAuth callback
export const handleOAuthCallback = async (req: AuthRequest, res: Response) => {
  const { provider } = req.params;
  const { code, state } = req.query;
  
  if (!code || !state || typeof code !== 'string' || typeof state !== 'string') {
    return res.status(400).json({ message: 'Invalid callback parameters' });
  }
  
  try {
    // Decode and verify state
    const decodedState = JSON.parse(Buffer.from(state, 'base64').toString());
    
    if (req.session.oauthState !== JSON.stringify(decodedState)) {
      return res.status(400).json({ message: 'Invalid state parameter' });
    }
    
    const { projectId } = decodedState;
    
    // Exchange code for tokens
    const tokens = await exchangeCodeForTokens(
      provider as 'googleAds' | 'facebookAds' | 'googleAnalytics',
      code
    );
    
    // Store connection in database
    if (!req.user) {
      return res.status(401).json({ message: 'Not authenticated' });
    }
    
    await storage.createDataSource({
      projectId: parseInt(projectId),
      type: provider === 'googleAds' ? 'google_ads' :
            provider === 'facebookAds' ? 'facebook_ads' : 'google_analytics',
      connectionInfo: tokens,
      createdById: req.user.id,
    });
    
    // Clear OAuth state from session
    delete req.session.oauthState;
    
    // Redirect to project page
    return res.redirect(`/projects/${projectId}/data-sources?connected=${provider}`);
  } catch (error) {
    console.error(`Error handling ${provider} OAuth callback:`, error);
    return res.status(500).json({ message: 'Error handling OAuth callback' });
  }
};

// Fetch data from the connected service
export const fetchConnectedData = async (dataSourceId: number) => {
  try {
    const dataSource = await storage.getDataSource(dataSourceId);
    if (!dataSource) {
      throw new Error('Data source not found');
    }
    
    // Here, you would use the appropriate API library to fetch data
    // based on the data source type and connection info
    
    switch (dataSource.type) {
      case 'google_ads':
        return await fetchGoogleAdsData(dataSource.connectionInfo);
      case 'facebook_ads':
        return await fetchFacebookAdsData(dataSource.connectionInfo);
      case 'google_analytics':
        return await fetchGoogleAnalyticsData(dataSource.connectionInfo);
      default:
        throw new Error(`Unsupported data source type: ${dataSource.type}`);
    }
  } catch (error) {
    console.error(`Error fetching data from source ${dataSourceId}:`, error);
    throw error;
  }
};

// Mock function for Google Ads API
async function fetchGoogleAdsData(connectionInfo: any) {
  // In a real implementation, you would use the Google Ads API client
  console.log('Fetching Google Ads data with tokens:', connectionInfo);
  
  // Return mock data for now
  return {
    data: [
      { date: '2023-01-01', spend: 100, impressions: 5000, clicks: 150 },
      { date: '2023-01-02', spend: 120, impressions: 5500, clicks: 170 },
      // More data...
    ],
    meta: {
      accountId: '123456789',
      currency: 'USD',
    },
  };
}

// Mock function for Facebook Ads API
async function fetchFacebookAdsData(connectionInfo: any) {
  // In a real implementation, you would use the Facebook Ads API client
  console.log('Fetching Facebook Ads data with tokens:', connectionInfo);
  
  // Return mock data for now
  return {
    data: [
      { date: '2023-01-01', spend: 200, impressions: 10000, reach: 8000 },
      { date: '2023-01-02', spend: 180, impressions: 9500, reach: 7800 },
      // More data...
    ],
    meta: {
      accountId: 'act_123456789',
      currency: 'USD',
    },
  };
}

// Mock function for Google Analytics API
async function fetchGoogleAnalyticsData(connectionInfo: any) {
  // In a real implementation, you would use the Google Analytics API client
  console.log('Fetching Google Analytics data with tokens:', connectionInfo);
  
  // Return mock data for now
  return {
    data: [
      { date: '2023-01-01', sessions: 1200, users: 980, pageviews: 3500 },
      { date: '2023-01-02', sessions: 1150, users: 950, pageviews: 3300 },
      // More data...
    ],
    meta: {
      propertyId: 'UA-123456-7',
      viewId: '12345678',
    },
  };
}
