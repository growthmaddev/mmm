import React from 'react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { useAuth } from "../hooks/useAuth";
import { User } from '@/lib/types';

interface UserProfileDropdownProps {
  mobileView?: boolean;
}

export default function UserProfileDropdown({
  mobileView = false,
}: UserProfileDropdownProps) {
  const { user, isAuthenticated } = useAuth();
  const userData = user as User | undefined;

  if (!isAuthenticated || !userData) {
    return (
      <Button 
        className={mobileView ? "w-full justify-start" : ""} 
        onClick={() => window.location.href = "/api/login"}
      >
        Log in
      </Button>
    );
  }

  const initials = userData?.firstName && userData?.lastName 
    ? `${userData.firstName[0]}${userData.lastName[0]}`
    : userData?.email 
      ? userData.email[0].toUpperCase() 
      : "U";

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" className="relative h-10 w-10 rounded-full">
          <Avatar>
            {userData.profileImageUrl ? (
              <AvatarImage src={userData.profileImageUrl} alt={userData.firstName || "User"} />
            ) : null}
            <AvatarFallback>{initials}</AvatarFallback>
          </Avatar>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuLabel>
          {userData.firstName && userData.lastName 
            ? `${userData.firstName} ${userData.lastName}` 
            : userData.email}
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={() => window.location.href = "/dashboard"}>
          Dashboard
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => window.location.href = "/api/logout"}>
          Log out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}