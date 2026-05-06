// src/app/components/take-recommendation/take-recommendation.component.ts
import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { UserService } from '../../../../services/user.service';

interface Book {
    title: string;
    author: string;
    genre: string;
    description: string;
    url: string;
    match_scores: {
        overall_match: number;
        notes_match: number;
        genre_match: number;
        feedback_match: number;
    };
}

interface RecommendationResponse {
    success: boolean;
    data: {
        book: Book;
        playlist: Array<{
            song: string;
            artist: string;
            year: number;
            popularity: number;
            danceability: number;
            energy: number;
            valence: number;
            tempo: number;
            genre: string;
        }>;
        match_scores: {
            overall_match: number;
            notes_match: number;
            genre_match: number;
            feedback_match: number;
        };
    };
    message?: string;
}

@Component({
  standalone: false,
    selector: 'app-take-recommendation',
    templateUrl: './take-recommendation.component.html',
    styleUrls: ['./take-recommendation.component.css']
})
export class TakeRecommendationComponent implements OnInit {
    recommendedBook: Book | null = null;
    recommendedPlaylist: { song: string, artist: string, year: number, popularity: number, danceability: number, energy: number, valence: number, tempo: number, genre: string }[] = [];

    constructor(private http: HttpClient, private userService: UserService) { }

    ngOnInit() {
        console.log("TakeRecommendationComponent initialized.");
        this.takeRec();
    }
    isLoading = true;
    takeRec() {
    this.isLoading = true;

    const username = this.userService.getUsername();

    if (!username) {
        console.error("No username found!");
        this.isLoading = false;
        return;
    }

    this.http.post<RecommendationResponse>('http://localhost:5000/recommend', { username }).subscribe({
        next: (response) => {
        console.log("Response received:", response);

        if (response.success && response.data) {
            this.recommendedBook = {
            ...response.data.book,
            match_scores: response.data.match_scores
            };

            this.recommendedPlaylist = response.data.playlist || [];

            console.log("Updated recommendedBook:", this.recommendedBook);
            console.log("Updated recommendedPlaylist:", this.recommendedPlaylist);
        } else {
            this.recommendedBook = null;
            this.recommendedPlaylist = [];
        }

        this.isLoading = false;
        },
        error: (error) => {
        console.error("API error:", error);
        this.recommendedBook = null;
        this.recommendedPlaylist = [];
        this.isLoading = false;
        }
    });
    }
}