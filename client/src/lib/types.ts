export interface User {
  id: string;
  email: string | null;
  firstName: string | null;
  lastName: string | null;
  profileImageUrl: string | null;
  organizationId: number | null;
  role: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface Organization {
  id: number;
  name: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface Project {
  id: number;
  name: string;
  description: string | null;
  organizationId: number;
  createdById: string;
  status: 'draft' | 'uploading_data' | 'configuring_model' | 'training' | 'completed' | 'error';
  dateRange: { startDate: string; endDate: string } | null;
  createdAt: Date;
  updatedAt: Date;
}

export interface DataSource {
  id: number;
  projectId: number;
  type: 'csv_upload' | 'google_ads' | 'facebook_ads' | 'google_analytics' | 'manual_entry';
  fileName: string | null;
  fileUrl: string | null;
  connectionInfo: any | null;
  dateColumn: string | null;
  metricColumns: string[] | null;
  channelColumns: { [key: string]: string } | null;
  controlColumns: { [key: string]: string } | null;
  createdById: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface Model {
  id: number;
  projectId: number;
  name: string;
  status: 'queued' | 'preprocessing' | 'training' | 'postprocessing' | 'completed' | 'error';
  progress: number;
  adstockSettings: any | null;
  saturationSettings: any | null;
  controlVariables: any | null;
  responseVariables: any | null;
  results: any | null;
  createdById: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface Channel {
  id: number;
  projectId: number;
  name: string;
  type: 'display' | 'search' | 'social' | 'email' | 'tv' | 'radio' | 'print' | 'ooh' | 'other';
  spend: { [date: string]: number } | null;
  createdAt: Date;
  updatedAt: Date;
}

export interface BudgetScenario {
  id: number;
  modelId: number;
  name: string;
  totalBudget: number;
  allocations: { [channelId: string]: number } | null;
  projectedResults: any | null;
  createdById: string;
  createdAt: Date;
  updatedAt: Date;
}