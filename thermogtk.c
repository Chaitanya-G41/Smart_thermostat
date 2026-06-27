// cp /mnt/user-data/uploads/Screenshot_2026-06-21_144121.png /tmp/orig.png
// # Read the C file from the context — it was provided inline, write it out
// cat > /home/claude/thermostat_gtk.c << 'CEOF'
/*
 * AI THERMOSTAT — GTK3 DASHBOARD  v3.2
 * Compile:
 *   gcc thermostat_gtk.c -o thermostat_gtk \
 *       $(pkg-config --cflags --libs gtk+-3.0) -lm
 */

#include <gtk/gtk.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_HISTORY       168
#define MAX_SCHEDULES      10
#define MAX_NAME_LEN       32
#define LEARNING_RATE      0.15
#define COMFORT_WEIGHT     0.6
#define ENERGY_WEIGHT      0.4
#define TEMP_MIN          10.0
#define TEMP_MAX          35.0
#define PREDICTION_WINDOW   6
#define OCCUPANCY_SETBACK  3.0
#define SAVE_FILE         "thermostat.dat"
#define VERSION           "3.2"

typedef struct {
    double temperature;
    int    hour;
    int    day;
    int    occupied;
    double user_setpoint;
    double outside_temp;
} HistoryRecord;

typedef struct {
    char   name[MAX_NAME_LEN];
    int    day_of_week;
    int    start_hour;
    int    end_hour;
    double target_temp;
    int    active;
} Schedule;

typedef struct {
    double learned_comfort[24];
    double confidence[24];
    double predicted[PREDICTION_WINDOW];

    double current_temp;
    double current_setpoint;
    double outside_temp;
    int    occupied;
    int    heating_on;
    int    cooling_on;

    HistoryRecord history[MAX_HISTORY];
    int           history_count;
    int           history_head;

    Schedule schedules[MAX_SCHEDULES];
    int      schedule_count;

    double total_energy_kwh;
    double session_energy_kwh;
    int    total_runtime_min;
    int    total_interactions;

    int    sim_hour;
    int    sim_day;
} Thermostat;

static Thermostat T;

static GtkWidget *window;
static GtkWidget *lbl_time,    *lbl_indoor,  *lbl_setpoint;
static GtkWidget *lbl_outside, *lbl_heating, *lbl_cooling;
static GtkWidget *lbl_occup,   *lbl_energy,  *lbl_interactions;
static GtkWidget *lbl_status_bar;
static GtkWidget *scale_setpoint;
static GtkWidget *drawing_chart;
static GtkWidget *drawing_comfort;
static GtkWidget *schedule_list;
static GtkWidget *lbl_sched_count;

static GtkWidget *dlg_name, *dlg_day, *dlg_start,
                 *dlg_end,  *dlg_temp, *dlg_active;

void   thermostat_init(void);
void   first_run_defaults(void);
int    save_state(void);
int    load_state(void);
void   learn_from_interaction(int hour, double setpoint);
void   predict_next_hours(void);
double optimise_setpoint(int hour);
double simulate_outside_temp(int hour);
void   update_hvac(void);
void   simulate_hour(void);
double get_schedule_setpoint(int day, int hour);
const char *day_name(int d);

void   build_ui(GtkApplication *app);
void   refresh_status(void);
void   refresh_schedules_list(void);
static gboolean on_tick(gpointer data);
static void     on_set_manual(GtkWidget *w, gpointer d);
static void     on_ai_setpoint(GtkWidget *w, gpointer d);
static void     on_toggle_occupancy(GtkWidget *w, gpointer d);
static void     on_advance_1h(GtkWidget *w, gpointer d);
static void     on_advance_6h(GtkWidget *w, gpointer d);
static void     on_advance_24h(GtkWidget *w, gpointer d);
static void     on_save(GtkWidget *w, gpointer d);
static void     on_add_schedule(GtkWidget *w, gpointer d);
static void     on_edit_schedule(GtkWidget *w, gpointer d);
static void     on_delete_schedule(GtkWidget *w, gpointer d);
static void     on_window_destroy(GtkWidget *w, gpointer d);
static gboolean on_draw_chart(GtkWidget *w, cairo_t *cr, gpointer d);
static gboolean on_draw_comfort(GtkWidget *w, cairo_t *cr, gpointer d);

int main(int argc, char **argv)
{
    thermostat_init();
    GtkApplication *app = gtk_application_new(
        "com.thermostat.ai", G_APPLICATION_DEFAULT_FLAGS);
    g_signal_connect(app, "activate", G_CALLBACK(build_ui), NULL);
    int status = g_application_run(G_APPLICATION(app), argc, argv);
    g_object_unref(app);
    return status;
}

void thermostat_init(void)
{
    if (load_state()) {
        T.session_energy_kwh = 0.0;
        predict_next_hours();
    } else {
        first_run_defaults();
        predict_next_hours();
    }
}

void first_run_defaults(void)
{
    int h;
    memset(&T, 0, sizeof(T));
    T.current_temp     = 19.0;
    T.current_setpoint = 21.0;
    T.outside_temp     = 10.0;
    T.occupied         = 1;
    T.sim_hour         = 8;
    T.sim_day          = 0;
    for (h = 0; h < 24; h++) {
        T.learned_comfort[h] = 21.0;
        T.confidence[h]      = 0.0;
    }
}

int save_state(void)
{
    FILE *f = fopen(SAVE_FILE, "wb");
    if (!f) return 0;
    unsigned int ver = 310;
    fwrite(&ver, sizeof(ver), 1, f);
    fwrite(&T,   sizeof(T),   1, f);
    fclose(f);
    return 1;
}

int load_state(void)
{
    FILE *f = fopen(SAVE_FILE, "rb");
    if (!f) return 0;
    unsigned int ver = 0;
    if (fread(&ver, sizeof(ver), 1, f) != 1 || ver != 310) {
        fclose(f); return 0;
    }
    size_t r = fread(&T, sizeof(T), 1, f);
    fclose(f);
    return (r == 1) ? 1 : 0;
}

void learn_from_interaction(int hour, double setpoint)
{
    double old  = T.learned_comfort[hour];
    double conf = T.confidence[hour];
    if (conf < 1.0)
        T.learned_comfort[hour] = setpoint;
    else
        T.learned_comfort[hour] = (1.0 - LEARNING_RATE) * old
                                  + LEARNING_RATE * setpoint;
    T.confidence[hour] = fmin(conf + 1.0, 200.0);
    T.total_interactions++;
}

void predict_next_hours(void)
{
    int i;
    double recent[6];
    int n = (T.history_count < 6) ? T.history_count : 6;
    for (i = 0; i < n; i++) {
        int idx = (T.history_head - 1 - i + MAX_HISTORY) % MAX_HISTORY;
        recent[i] = T.history[idx].user_setpoint;
    }
    double trend = (n > 1) ? (recent[0] - recent[n-1]) / (double)n : 0.0;
    for (i = 0; i < PREDICTION_WINDOW; i++) {
        int h = (T.sim_hour + i) % 24;
        double model_val   = T.learned_comfort[h];
        double conf        = T.confidence[h];
        double conf_factor = fmin(conf / 20.0, 1.0);
        double trend_adj   = model_val + trend * (double)(i + 1);
        T.predicted[i]     = conf_factor * model_val
                            + (1.0 - conf_factor) * trend_adj;
        if (T.predicted[i] < TEMP_MIN) T.predicted[i] = TEMP_MIN;
        if (T.predicted[i] > TEMP_MAX) T.predicted[i] = TEMP_MAX;
    }
}

double optimise_setpoint(int hour)
{
    double pref    = T.learned_comfort[hour];
    double outside = simulate_outside_temp(hour);
    if (!T.occupied) pref -= OCCUPANCY_SETBACK;
    double w   = ENERGY_WEIGHT / (COMFORT_WEIGHT * 4.0 + 1e-9);
    double opt = (pref + w * outside) / (1.0 + w);
    if (opt < TEMP_MIN) opt = TEMP_MIN;
    if (opt > TEMP_MAX) opt = TEMP_MAX;
    return opt;
}

double simulate_outside_temp(int hour)
{
    return 10.0 + 5.0 * sin(M_PI * (double)(hour - 4) / 12.0);
}

void update_hvac(void)
{
    double delta    = T.current_setpoint - T.current_temp;
    double deadband = 0.3;
    T.heating_on = (delta >  deadband) ? 1 : 0;
    T.cooling_on = (delta < -deadband) ? 1 : 0;
    if (T.heating_on) {
        double power = fmin(fabs(delta) * 0.3, 0.8);
        T.current_temp       += power;
        T.session_energy_kwh += (power / 0.8) * (1.5 / 60.0);
        T.total_energy_kwh   += (power / 0.8) * (1.5 / 60.0);
        T.total_runtime_min++;
    } else if (T.cooling_on) {
        double power = fmin(fabs(delta) * 0.25, 0.6);
        T.current_temp       -= power;
        T.session_energy_kwh += (power / 0.6) * (1.2 / 60.0);
        T.total_energy_kwh   += (power / 0.6) * (1.2 / 60.0);
        T.total_runtime_min++;
    } else {
        T.current_temp += 0.05 * (T.outside_temp - T.current_temp);
    }
}

void simulate_hour(void)
{
    int i;
    T.outside_temp = simulate_outside_temp(T.sim_hour);
    double sched   = get_schedule_setpoint(T.sim_day, T.sim_hour);
    T.current_setpoint = (sched > 0.0) ? sched : optimise_setpoint(T.sim_hour);
    for (i = 0; i < 60; i++) update_hvac();

    HistoryRecord *rec = &T.history[T.history_head];
    rec->temperature   = T.current_temp;
    rec->hour          = T.sim_hour;
    rec->day           = T.sim_day;
    rec->occupied      = T.occupied;
    rec->user_setpoint = T.current_setpoint;
    rec->outside_temp  = T.outside_temp;
    T.history_head = (T.history_head + 1) % MAX_HISTORY;
    if (T.history_count < MAX_HISTORY) T.history_count++;

    T.sim_hour = (T.sim_hour + 1) % 24;
    if (T.sim_hour == 0) T.sim_day = (T.sim_day + 1) % 7;
    predict_next_hours();
}

double get_schedule_setpoint(int day, int hour)
{
    int i;
    for (i = 0; i < T.schedule_count; i++) {
        Schedule *s = &T.schedules[i];
        if (!s->active) continue;
        int dm = (s->day_of_week == -1 || s->day_of_week == day);
        int tm;
        if (s->start_hour < s->end_hour)
            tm = (hour >= s->start_hour && hour < s->end_hour);
        else
            tm = (hour >= s->start_hour || hour < s->end_hour);
        if (dm && tm) return s->target_temp;
    }
    return 0.0;
}

const char *day_name(int d)
{
    static const char *n[] = {
        "Monday","Tuesday","Wednesday","Thursday",
        "Friday","Saturday","Sunday"
    };
    return (d >= 0 && d <= 6) ? n[d] : "?";
}

/* ============================================================
 *  CAIRO — TEMPERATURE CHART
 * ============================================================ */
static gboolean on_draw_chart(GtkWidget *widget, cairo_t *cr, gpointer data)
{
    (void)data;
    int W = gtk_widget_get_allocated_width(widget);
    int H = gtk_widget_get_allocated_height(widget);
    int pl = 52, pr = 16, pt = 28, pb = 36;
    int pw = W - pl - pr;
    int ph = H - pt - pb;

    cairo_set_source_rgb(cr, 0.10, 0.10, 0.13);
    cairo_rectangle(cr, 0, 0, W, H);
    cairo_fill(cr);

    cairo_set_source_rgba(cr, 1,1,1, 0.06);
    cairo_set_line_width(cr, 1);
    cairo_rectangle(cr, pl, pt, pw, ph);
    cairo_stroke(cr);

    double yticks[] = {10, 15, 20, 25, 30, 35};
    int ti;
    cairo_select_font_face(cr, "monospace",
        CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, 12);

    for (ti = 0; ti < 6; ti++) {
        double t  = yticks[ti];
        double yy = pt + ph - (t - TEMP_MIN) / (TEMP_MAX - TEMP_MIN) * ph;

        cairo_set_source_rgba(cr, 1,1,1, 0.08);
        cairo_set_line_width(cr, 1);
        cairo_move_to(cr, pl, yy);
        cairo_line_to(cr, pl + pw, yy);
        cairo_stroke(cr);

        cairo_set_source_rgba(cr, 1,1,1, 0.6);
        char buf[8]; snprintf(buf, sizeof(buf), "%2.0f°", t);
        cairo_move_to(cr, 4, yy + 5);
        cairo_show_text(cr, buf);
    }

    {
        double y_top = pt + ph - (23.0 - TEMP_MIN) / (TEMP_MAX - TEMP_MIN) * ph;
        double y_bot = pt + ph - (18.0 - TEMP_MIN) / (TEMP_MAX - TEMP_MIN) * ph;
        cairo_set_source_rgba(cr, 0.27, 0.78, 0.64, 0.07);
        cairo_rectangle(cr, pl, y_top, pw, y_bot - y_top);
        cairo_fill(cr);
    }

    int show = (T.history_count < 24) ? T.history_count : 24;

    if (show == 0 && T.history_count == 0) {
        cairo_set_source_rgba(cr, 1,1,1, 0.30);
        cairo_set_font_size(cr, 14);
        cairo_move_to(cr, pl + pw/2 - 130, pt + ph/2);
        cairo_show_text(cr, "Simulate hours to populate chart");
        return FALSE;
    }

    double hist_w  = pw * 0.65;
    double fore_w  = pw * 0.35;
    double fore_x0 = pl + hist_w;

    double xs[MAX_HISTORY + PREDICTION_WINDOW + 1];
    double ys[MAX_HISTORY + PREDICTION_WINDOW + 1];
    double ss[MAX_HISTORY + PREDICTION_WINDOW + 1];

    int i;
    for (i = 0; i < show; i++) {
        int idx = (T.history_head - show + i + MAX_HISTORY) % MAX_HISTORY;
        HistoryRecord *r = &T.history[idx];
        double x_frac = (show > 1) ? (double)i / (double)(show - 1) : 1.0;
        xs[i] = pl + x_frac * hist_w;
        ys[i] = pt + ph - (r->temperature  - TEMP_MIN)/(TEMP_MAX-TEMP_MIN)*ph;
        ss[i] = pt + ph - (r->user_setpoint - TEMP_MIN)/(TEMP_MAX-TEMP_MIN)*ph;
    }

    double anchor_x = (show > 0) ? xs[show-1] : fore_x0;
    double anchor_y = (show > 0) ? ys[show-1]
                    : pt + ph - (T.current_temp - TEMP_MIN)/(TEMP_MAX-TEMP_MIN)*ph;

    for (i = 0; i < PREDICTION_WINDOW; i++) {
        double x_frac = (double)(i+1) / (double)PREDICTION_WINDOW;
        xs[show+i] = fore_x0 + x_frac * fore_w;
        ys[show+i] = pt + ph - (T.predicted[i] - TEMP_MIN)/(TEMP_MAX-TEMP_MIN)*ph;
        ss[show+i] = ys[show+i];
    }

    cairo_set_source_rgba(cr, 1,1,1, 0.12);
    cairo_set_line_width(cr, 1);
    double dash[] = {5, 5};
    cairo_set_dash(cr, dash, 2, 0);
    cairo_move_to(cr, fore_x0, pt + 4);
    cairo_line_to(cr, fore_x0, pt + ph - 4);
    cairo_stroke(cr);
    cairo_set_dash(cr, NULL, 0, 0);

    if (show >= 1) {
        cairo_set_source_rgba(cr, 0.94, 0.63, 0.15, 0.70);
        cairo_set_line_width(cr, 2.0);
        double d2[] = {6, 4};
        cairo_set_dash(cr, d2, 2, 0);
        cairo_move_to(cr, xs[0], ss[0]);
        for (i = 1; i < show; i++) cairo_line_to(cr, xs[i], ss[i]);
        cairo_stroke(cr);
        cairo_set_dash(cr, NULL, 0, 0);
    }

    if (show >= 1) {
        cairo_set_source_rgba(cr, 0.27, 0.78, 0.64, 1.0);
        cairo_set_line_width(cr, 2.5);
        cairo_move_to(cr, xs[0], ys[0]);
        for (i = 1; i < show; i++) cairo_line_to(cr, xs[i], ys[i]);
        cairo_stroke(cr);
    }

    if (PREDICTION_WINDOW > 0) {
        cairo_set_source_rgba(cr, 0.27, 0.78, 0.64, 0.5);
        cairo_set_line_width(cr, 1.5);
        double d3[] = {5, 4};
        cairo_set_dash(cr, d3, 2, 0);
        cairo_move_to(cr, anchor_x, anchor_y);
        for (i = 0; i < PREDICTION_WINDOW; i++)
            cairo_line_to(cr, xs[show+i], ys[show+i]);
        cairo_stroke(cr);
        cairo_set_dash(cr, NULL, 0, 0);
    }

    for (i = 0; i < show; i++) {
        cairo_set_source_rgba(cr, 0.27, 0.78, 0.64, 0.55);
        cairo_arc(cr, xs[i], ys[i], 3.5, 0, 2*M_PI);
        cairo_fill(cr);
    }

    cairo_set_font_size(cr, 11);
    for (i = 0; i < PREDICTION_WINDOW; i++) {
        cairo_set_source_rgba(cr, 0.27, 0.78, 0.64, 0.65);
        cairo_arc(cr, xs[show+i], ys[show+i], 4.5, 0, 2*M_PI);
        cairo_fill(cr);
        cairo_set_source_rgba(cr, 1,1,1, 0.7);
        char buf[8];
        snprintf(buf, sizeof(buf), "%.1f", T.predicted[i]);
        cairo_move_to(cr, xs[show+i] - 12, ys[show+i] - 8);
        cairo_show_text(cr, buf);
    }

    if (show > 0) {
        cairo_set_source_rgba(cr, 0.27, 0.78, 0.64, 0.22);
        cairo_arc(cr, xs[show-1], ys[show-1], 9, 0, 2*M_PI);
        cairo_fill(cr);
        cairo_set_source_rgb(cr, 0.27, 0.78, 0.64);
        cairo_arc(cr, xs[show-1], ys[show-1], 5.5, 0, 2*M_PI);
        cairo_fill(cr);
    }

    cairo_set_source_rgba(cr, 1,1,1, 0.45);
    cairo_set_font_size(cr, 12);
    cairo_move_to(cr, pl + 4, pt + ph + 22);
    cairo_show_text(cr, "← history");
    cairo_move_to(cr, fore_x0 + 4, pt + ph + 22);
    cairo_show_text(cr, "forecast →");

    double ly = pt + 12;
    cairo_set_source_rgba(cr, 0.27, 0.78, 0.64, 1.0);
    cairo_set_line_width(cr, 2.5);
    cairo_move_to(cr, pl + 8, ly); cairo_line_to(cr, pl + 28, ly);
    cairo_stroke(cr);
    cairo_set_source_rgba(cr, 1,1,1, 0.75);
    cairo_set_font_size(cr, 12);
    cairo_move_to(cr, pl + 32, ly + 5);
    cairo_show_text(cr, "Indoor");

    cairo_set_source_rgba(cr, 0.94, 0.63, 0.15, 0.85);
    cairo_set_line_width(cr, 2.0);
    double d4[] = {5, 3};
    cairo_set_dash(cr, d4, 2, 0);
    cairo_move_to(cr, pl + 100, ly); cairo_line_to(cr, pl + 120, ly);
    cairo_stroke(cr);
    cairo_set_dash(cr, NULL, 0, 0);
    cairo_set_source_rgba(cr, 1,1,1, 0.75);
    cairo_move_to(cr, pl + 124, ly + 5);
    cairo_show_text(cr, "Setpoint");

    return FALSE;
}

/* ============================================================
 *  CAIRO — COMFORT PROFILE
 * ============================================================ */
static gboolean on_draw_comfort(GtkWidget *widget, cairo_t *cr, gpointer data)
{
    (void)data;
    int W = gtk_widget_get_allocated_width(widget);
    int H = gtk_widget_get_allocated_height(widget);
    int pl = 46, pr = 12, pt = 24, pb = 36;
    int pw = W - pl - pr;
    int ph = H - pt - pb;

    cairo_set_source_rgb(cr, 0.10, 0.10, 0.13);
    cairo_rectangle(cr, 0, 0, W, H);
    cairo_fill(cr);

    double bar_w = (double)pw / 24.0;
    int hr;

    for (hr = 0; hr < 24; hr++) {
        double temp = T.learned_comfort[hr];
        double conf = T.confidence[hr];
        double norm = (temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN);
        double bh   = norm * ph;
        double bx   = pl + hr * bar_w;
        double by   = pt + ph - bh;

        if (conf < 1.0) {
            double def_norm = (21.0 - TEMP_MIN) / (TEMP_MAX - TEMP_MIN);
            double def_bh   = def_norm * ph;
            cairo_set_source_rgba(cr, 0.32, 0.32, 0.36, 0.40);
            cairo_rectangle(cr, bx+1, pt + ph - def_bh, bar_w-2, def_bh);
            cairo_fill(cr);
        } else {
            double alpha = fmin(0.5 + conf / 40.0, 1.0);
            double r_c = norm * 0.94 + (1.0-norm) * 0.22;
            double g_c = norm * 0.42 + (1.0-norm) * 0.62;
            double b_c = norm * 0.08 + (1.0-norm) * 0.88;

            cairo_set_source_rgba(cr, r_c, g_c, b_c, alpha);
            cairo_rectangle(cr, bx+1, by, bar_w-2, bh);
            cairo_fill(cr);

            cairo_set_source_rgba(cr, 1,1,1, 0.14);
            cairo_rectangle(cr, bx+1, by, bar_w-2, 4);
            cairo_fill(cr);

            if (bh > 28) {
                cairo_select_font_face(cr, "monospace",
                    CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
                cairo_set_font_size(cr, 11);
                cairo_set_source_rgba(cr, 1,1,1, 0.9);
                char tbuf[8]; snprintf(tbuf, sizeof(tbuf), "%.0f°", temp);
                cairo_move_to(cr, bx + bar_w/2 - 9, by + 16);
                cairo_show_text(cr, tbuf);
            }

            int dots = (int)fmin(conf, 5);
            int di;
            for (di = 0; di < dots; di++) {
                cairo_set_source_rgba(cr, 1,1,1, 0.55);
                cairo_arc(cr, bx + bar_w/2 - dots*4 + di*8,
                          pt + ph - 6, 2.5, 0, 2*M_PI);
                cairo_fill(cr);
            }
        }

        if (hr == T.sim_hour) {
            cairo_set_source_rgba(cr, 1, 1, 1, 0.85);
            cairo_set_line_width(cr, 2.5);
            cairo_rectangle(cr, bx+1, pt, bar_w-2, ph);
            cairo_stroke(cr);
            cairo_set_line_width(cr, 1);
        }

        cairo_set_source_rgba(cr, 0,0,0, 0.4);
        cairo_set_line_width(cr, 0.5);
        cairo_move_to(cr, bx + bar_w, pt);
        cairo_line_to(cr, bx + bar_w, pt + ph);
        cairo_stroke(cr);
    }

    cairo_select_font_face(cr, "monospace",
        CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, 11);
    double yt[] = {15, 20, 25, 30};
    int ti;
    for (ti = 0; ti < 4; ti++) {
        double yy = pt + ph - (yt[ti]-TEMP_MIN)/(TEMP_MAX-TEMP_MIN)*ph;
        cairo_set_source_rgba(cr, 1,1,1, 0.09);
        cairo_set_line_width(cr, 1);
        cairo_move_to(cr, pl, yy);
        cairo_line_to(cr, pl+pw, yy);
        cairo_stroke(cr);
        cairo_set_source_rgba(cr, 1,1,1, 0.55);
        char buf[6]; snprintf(buf, sizeof(buf), "%2.0f°", yt[ti]);
        cairo_move_to(cr, 4, yy+4);
        cairo_show_text(cr, buf);
    }

    cairo_set_source_rgba(cr, 1,1,1, 0.55);
    cairo_set_font_size(cr, 11);
    for (hr = 0; hr < 24; hr += 3) {
        double bx = pl + hr * bar_w;
        char buf[6]; snprintf(buf, sizeof(buf), "%02d", hr);
        cairo_move_to(cr, bx + bar_w/2 - 8, H - 10);
        cairo_show_text(cr, buf);
    }

    cairo_set_source_rgba(cr, 1,1,1, 0.35);
    cairo_set_font_size(cr, 11);
    cairo_move_to(cr, pl + 6, pt - 7);
    cairo_show_text(cr, "No data");
    cairo_set_source_rgba(cr, 0.22, 0.62, 0.88, 0.9);
    cairo_move_to(cr, pl + 76, pt - 7);
    cairo_show_text(cr, "Cold");
    cairo_set_source_rgba(cr, 0.94, 0.42, 0.08, 0.9);
    cairo_move_to(cr, pl + 122, pt - 7);
    cairo_show_text(cr, "Warm");

    return FALSE;
}

/* ============================================================
 *  UI REFRESH
 * ============================================================ */
void refresh_status(void)
{
    char buf[160];

    snprintf(buf, sizeof(buf), "%s  %02d:00", day_name(T.sim_day), T.sim_hour);
    gtk_label_set_text(GTK_LABEL(lbl_time), buf);

    snprintf(buf, sizeof(buf), "%.1f °C", T.current_temp);
    gtk_label_set_text(GTK_LABEL(lbl_indoor), buf);

    snprintf(buf, sizeof(buf), "%.1f °C", T.current_setpoint);
    gtk_label_set_text(GTK_LABEL(lbl_setpoint), buf);

    snprintf(buf, sizeof(buf), "%.1f °C", T.outside_temp);
    gtk_label_set_text(GTK_LABEL(lbl_outside), buf);

    if (T.heating_on) {
        gtk_label_set_markup(GTK_LABEL(lbl_heating),
            "<span foreground='#FF6644' size='large'>🔥 HEATING ON</span>");
    } else {
        gtk_label_set_markup(GTK_LABEL(lbl_heating),
            "<span foreground='#888899' size='large'>— Heating off</span>");
    }

    if (T.cooling_on) {
        gtk_label_set_markup(GTK_LABEL(lbl_cooling),
            "<span foreground='#44AAFF' size='large'>❄ COOLING ON</span>");
    } else {
        gtk_label_set_markup(GTK_LABEL(lbl_cooling),
            "<span foreground='#888899' size='large'>— Cooling off</span>");
    }

    if (T.occupied) {
        gtk_label_set_markup(GTK_LABEL(lbl_occup),
            "<span foreground='#45e0b8' size='large'>● Occupied</span>");
    } else {
        gtk_label_set_markup(GTK_LABEL(lbl_occup),
            "<span foreground='#EF9F27' size='large'>● Unoccupied  (–3°C setback)</span>");
    }

    snprintf(buf, sizeof(buf), "%.3f kWh total\n%.3f kWh session",
             T.total_energy_kwh, T.session_energy_kwh);
    gtk_label_set_text(GTK_LABEL(lbl_energy), buf);

    snprintf(buf, sizeof(buf), "%d", T.total_interactions);
    gtk_label_set_text(GTK_LABEL(lbl_interactions), buf);

    gtk_range_set_value(GTK_RANGE(scale_setpoint), T.current_setpoint);

    gtk_widget_queue_draw(drawing_chart);
    gtk_widget_queue_draw(drawing_comfort);
}

/* ============================================================
 *  SCHEDULE LIST
 * ============================================================ */
void refresh_schedules_list(void)
{
    GList *children = gtk_container_get_children(GTK_CONTAINER(schedule_list));
    GList *c;
    for (c = children; c; c = c->next)
        gtk_widget_destroy(GTK_WIDGET(c->data));
    g_list_free(children);

    int i;
    for (i = 0; i < T.schedule_count; i++) {
        Schedule *s = &T.schedules[i];

        GtkWidget *row = gtk_list_box_row_new();

        GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 14);
        gtk_container_set_border_width(GTK_CONTAINER(hbox), 14);

        GtkWidget *dot = gtk_label_new(s->active ? "●" : "○");
        GtkStyleContext *dctx = gtk_widget_get_style_context(dot);
        gtk_style_context_add_class(dctx, s->active ? "dot-on" : "dot-off");
        gtk_box_pack_start(GTK_BOX(hbox), dot, FALSE, FALSE, 0);

        GtkWidget *lname = gtk_label_new(s->name);
        gtk_label_set_xalign(GTK_LABEL(lname), 0);
        gtk_widget_set_size_request(lname, 170, -1);
        GtkStyleContext *nctx = gtk_widget_get_style_context(lname);
        gtk_style_context_add_class(nctx, "sched-name");
        gtk_box_pack_start(GTK_BOX(hbox), lname, FALSE, FALSE, 0);

        char daybuf[16];
        snprintf(daybuf, sizeof(daybuf), "%s",
                 s->day_of_week == -1 ? "Every day" : day_name(s->day_of_week));
        GtkWidget *lday = gtk_label_new(daybuf);
        gtk_label_set_xalign(GTK_LABEL(lday), 0);
        gtk_widget_set_size_request(lday, 110, -1);
        GtkStyleContext *dctx2 = gtk_widget_get_style_context(lday);
        gtk_style_context_add_class(dctx2, "sched-meta");
        gtk_box_pack_start(GTK_BOX(hbox), lday, FALSE, FALSE, 0);

        char timebuf[24];
        snprintf(timebuf, sizeof(timebuf), "%02d:00 – %02d:00",
                 s->start_hour, s->end_hour);
        GtkWidget *ltime = gtk_label_new(timebuf);
        gtk_label_set_xalign(GTK_LABEL(ltime), 0);
        gtk_widget_set_size_request(ltime, 120, -1);
        GtkStyleContext *tctx = gtk_widget_get_style_context(ltime);
        gtk_style_context_add_class(tctx, "sched-meta");
        gtk_box_pack_start(GTK_BOX(hbox), ltime, FALSE, FALSE, 0);

        char tempbuf[16];
        snprintf(tempbuf, sizeof(tempbuf), "%.1f °C", s->target_temp);
        GtkWidget *ltemp = gtk_label_new(tempbuf);
        GtkStyleContext *tmpctx = gtk_widget_get_style_context(ltemp);
        gtk_style_context_add_class(tmpctx, "temp-badge");
        gtk_box_pack_end(GTK_BOX(hbox), ltemp, FALSE, FALSE, 10);

        gtk_container_add(GTK_CONTAINER(row), hbox);
        gtk_list_box_insert(GTK_LIST_BOX(schedule_list), row, -1);
    }

    if (T.schedule_count == 0) {
        GtkWidget *row = gtk_list_box_row_new();
        GtkWidget *lbl = gtk_label_new("No schedules yet — click + Add to create one");
        gtk_label_set_xalign(GTK_LABEL(lbl), 0.5);
        GtkStyleContext *ctx = gtk_widget_get_style_context(lbl);
        gtk_style_context_add_class(ctx, "empty-hint");
        gtk_container_set_border_width(GTK_CONTAINER(row), 24);
        gtk_container_add(GTK_CONTAINER(row), lbl);
        gtk_list_box_insert(GTK_LIST_BOX(schedule_list), row, -1);
    }

    char cbuf[40];
    snprintf(cbuf, sizeof(cbuf), "Schedules  (%d / %d)",
             T.schedule_count, MAX_SCHEDULES);
    gtk_label_set_text(GTK_LABEL(lbl_sched_count), cbuf);

    gtk_widget_show_all(schedule_list);
}

/* ============================================================
 *  CALLBACKS
 * ============================================================ */
static gboolean on_tick(gpointer data)
{
    (void)data;
    update_hvac();
    refresh_status();
    return G_SOURCE_CONTINUE;
}

static void set_status(const char *msg)
{
    gtk_label_set_text(GTK_LABEL(lbl_status_bar), msg);
}

static void on_set_manual(GtkWidget *w, gpointer d)
{
    (void)w; (void)d;
    double val = gtk_range_get_value(GTK_RANGE(scale_setpoint));
    learn_from_interaction(T.sim_hour, val);
    T.current_setpoint = val;
    update_hvac();
    refresh_status();
    char buf[80];
    snprintf(buf, sizeof(buf),
             "Setpoint %.1f °C applied and learned for %02d:00",
             val, T.sim_hour);
    set_status(buf);
}

static void on_ai_setpoint(GtkWidget *w, gpointer d)
{
    (void)w; (void)d;
    T.current_setpoint = optimise_setpoint(T.sim_hour);
    update_hvac();
    refresh_status();
    char buf[80];
    snprintf(buf, sizeof(buf), "AI suggests %.1f °C%s",
             T.current_setpoint,
             T.occupied ? "" : "  (occupancy setback included)");
    set_status(buf);
}

static void on_toggle_occupancy(GtkWidget *w, gpointer d)
{
    (void)w; (void)d;
    T.occupied = !T.occupied;
    refresh_status();
    set_status(T.occupied ? "Room occupied"
                          : "Room unoccupied — 3 °C setback active");
}

static void advance_hours(int n)
{
    int i;
    for (i = 0; i < n; i++) simulate_hour();
    refresh_status();
    char buf[64];
    snprintf(buf, sizeof(buf), "Advanced %d hour(s)  →  %s %02d:00",
             n, day_name(T.sim_day), T.sim_hour);
    set_status(buf);
}

static void on_advance_1h(GtkWidget *w, gpointer d)  { (void)w;(void)d; advance_hours(1);  }
static void on_advance_6h(GtkWidget *w, gpointer d)  { (void)w;(void)d; advance_hours(6);  }
static void on_advance_24h(GtkWidget *w, gpointer d) { (void)w;(void)d; advance_hours(24); }

static void on_save(GtkWidget *w, gpointer d)
{
    (void)w; (void)d;
    set_status(save_state() ? "Data saved to thermostat.dat"
                            : "ERROR: save failed");
}

static void on_window_destroy(GtkWidget *w, gpointer d)
{
    (void)w; (void)d;
    save_state();
}

static GtkWidget *make_schedule_dialog(GtkWindow *parent,
                                       const char *title,
                                       Schedule   *prefill)
{
    GtkWidget *dlg = gtk_dialog_new_with_buttons(
        title, parent,
        GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT,
        "_Cancel", GTK_RESPONSE_CANCEL,
        "_OK",     GTK_RESPONSE_OK,
        NULL);

    GtkWidget *content = gtk_dialog_get_content_area(GTK_DIALOG(dlg));
    GtkWidget *grid    = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(grid), 14);
    gtk_grid_set_column_spacing(GTK_GRID(grid), 20);
    gtk_container_set_border_width(GTK_CONTAINER(grid), 24);
    gtk_container_add(GTK_CONTAINER(content), grid);

    #define ADD_ROW(r, ltext, widget) \
        { GtkWidget *_l = gtk_label_new(ltext); \
          gtk_label_set_xalign(GTK_LABEL(_l), 1.0); \
          gtk_grid_attach(GTK_GRID(grid), _l, 0, r, 1, 1); \
          gtk_grid_attach(GTK_GRID(grid), widget, 1, r, 1, 1); }

    dlg_name = gtk_entry_new();
    gtk_entry_set_max_length(GTK_ENTRY(dlg_name), MAX_NAME_LEN - 1);
    gtk_widget_set_size_request(dlg_name, 220, -1);
    ADD_ROW(0, "Name:", dlg_name)

    dlg_day = gtk_spin_button_new_with_range(-1, 6, 1);
    gtk_widget_set_tooltip_text(dlg_day, "-1 = every day");
    ADD_ROW(1, "Day (-1=every day):", dlg_day)

    dlg_start = gtk_spin_button_new_with_range(0, 23, 1);
    ADD_ROW(2, "Start hour:", dlg_start)

    dlg_end = gtk_spin_button_new_with_range(0, 23, 1);
    ADD_ROW(3, "End hour:", dlg_end)

    dlg_temp = gtk_spin_button_new_with_range(TEMP_MIN, TEMP_MAX, 0.5);
    gtk_spin_button_set_digits(GTK_SPIN_BUTTON(dlg_temp), 1);
    ADD_ROW(4, "Target temp (°C):", dlg_temp)

    dlg_active = gtk_check_button_new_with_label("Active");
    gtk_grid_attach(GTK_GRID(grid), dlg_active, 1, 5, 1, 1);

    if (prefill) {
        gtk_entry_set_text(GTK_ENTRY(dlg_name), prefill->name);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(dlg_day),   prefill->day_of_week);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(dlg_start), prefill->start_hour);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(dlg_end),   prefill->end_hour);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(dlg_temp),  prefill->target_temp);
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(dlg_active), prefill->active);
    } else {
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(dlg_day),   -1);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(dlg_start),  6);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(dlg_end),    9);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(dlg_temp),  21.0);
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(dlg_active), TRUE);
    }

    gtk_widget_show_all(dlg);
    #undef ADD_ROW
    return dlg;
}

static void read_schedule_dialog(Schedule *s)
{
    strncpy(s->name,
            gtk_entry_get_text(GTK_ENTRY(dlg_name)), MAX_NAME_LEN - 1);
    s->name[MAX_NAME_LEN - 1] = '\0';
    if (s->name[0] == '\0') strncpy(s->name, "Unnamed", MAX_NAME_LEN - 1);
    s->day_of_week = (int)gtk_spin_button_get_value(GTK_SPIN_BUTTON(dlg_day));
    s->start_hour  = (int)gtk_spin_button_get_value(GTK_SPIN_BUTTON(dlg_start));
    s->end_hour    = (int)gtk_spin_button_get_value(GTK_SPIN_BUTTON(dlg_end));
    s->target_temp = gtk_spin_button_get_value(GTK_SPIN_BUTTON(dlg_temp));
    s->active      = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(dlg_active));
}

static void on_add_schedule(GtkWidget *w, gpointer d)
{
    (void)w; (void)d;
    if (T.schedule_count >= MAX_SCHEDULES) {
        set_status("Schedule list full (max 10)");
        return;
    }
    GtkWidget *dlg = make_schedule_dialog(GTK_WINDOW(window), "Add Schedule", NULL);
    if (gtk_dialog_run(GTK_DIALOG(dlg)) == GTK_RESPONSE_OK) {
        read_schedule_dialog(&T.schedules[T.schedule_count]);
        T.schedule_count++;
        refresh_schedules_list();
        set_status("Schedule added");
    }
    gtk_widget_destroy(dlg);
}

static void on_edit_schedule(GtkWidget *w, gpointer d)
{
    (void)w; (void)d;
    GtkListBoxRow *row = gtk_list_box_get_selected_row(
        GTK_LIST_BOX(schedule_list));
    if (!row || T.schedule_count == 0) {
        set_status("Select a schedule row to edit");
        return;
    }
    int idx = gtk_list_box_row_get_index(row);
    if (idx < 0 || idx >= T.schedule_count) return;

    GtkWidget *dlg = make_schedule_dialog(GTK_WINDOW(window),
                                          "Edit Schedule", &T.schedules[idx]);
    if (gtk_dialog_run(GTK_DIALOG(dlg)) == GTK_RESPONSE_OK) {
        read_schedule_dialog(&T.schedules[idx]);
        refresh_schedules_list();
        set_status("Schedule updated");
    }
    gtk_widget_destroy(dlg);
}

static void on_delete_schedule(GtkWidget *w, gpointer d)
{
    (void)w; (void)d;
    GtkListBoxRow *row = gtk_list_box_get_selected_row(
        GTK_LIST_BOX(schedule_list));
    if (!row || T.schedule_count == 0) {
        set_status("Select a schedule row to delete");
        return;
    }
    int idx = gtk_list_box_row_get_index(row);
    if (idx < 0 || idx >= T.schedule_count) return;

    char msg[64];
    snprintf(msg, sizeof(msg), "Deleted: %s", T.schedules[idx].name);
    int i;
    for (i = idx; i < T.schedule_count - 1; i++)
        T.schedules[i] = T.schedules[i + 1];
    T.schedule_count--;
    refresh_schedules_list();
    set_status(msg);
}

/* ============================================================
 *  CSS  — v3.2 overhaul
 *  Key fixes vs v3.1:
 *    • Global font size bumped to 15px
 *    • Sidebar wider, stat-card text is BLACK (#111) on light
 *      backgrounds — actually we keep dark theme but make text
 *      explicitly bright (#e8e8f0) at 15px
 *    • Tab 3 (Schedules) gets explicit dark bg + dark text on rows
 *    • Buttons get more padding and 14px font
 *    • Sidebar value labels 16px, indoor big-temp 36px
 * ============================================================ */
static void apply_css(void)
{
    GtkCssProvider *css = gtk_css_provider_new();
    const char *style =

    /* === Global reset — everything inherits 15px dark-on-dark === */
    "* { font-family: 'Segoe UI', 'Ubuntu', 'DejaVu Sans', sans-serif;"
    "    font-size: 15px; color: #e0e0ee; }"

    /* === Window === */
    "window { background: #16161f; }"

    /* === Notebook tabs === */
    "notebook > header { background: #0e0e16; border-bottom: 2px solid #28283a; }"
    "notebook > header > tabs > tab {"
    "  background: transparent; color: #8888aa; padding: 12px 28px;"
    "  font-size: 15px; border: none; border-bottom: 3px solid transparent; }"
    "notebook > header > tabs > tab:checked {"
    "  color: #45e0b8; border-bottom: 3px solid #45e0b8;"
    "  background: transparent; font-weight: bold; }"
    "notebook > header > tabs > tab:hover:not(:checked) { color: #ccccdd; }"
    "notebook { background: #16161f; }"

    /* === Sidebar === */
    ".sidebar { background: #0e0e16; border-right: 1px solid #28283a; }"

    /* === Stat cards === */
    ".stat-card {"
    "  background: #1c1c28; border-radius: 10px;"
    "  padding: 14px 16px; margin: 4px 0;"
    "  border: 1px solid #2c2c42; }"
    ".stat-card-label { font-size: 12px; color: #9090aa; margin-bottom: 5px; }"
    ".stat-card-val   { font-size: 16px; color: #dde0f0; font-weight: bold; }"
    ".big-temp { font-size: 36px; font-weight: bold; color: #EF9F27; }"

    /* === HVAC labels === */
    ".hvac-label { font-size: 14px; padding: 3px 0; }"

    /* === Buttons — larger, more breathing room === */
    "button {"
    "  background: #23233a; color: #d0d0e8;"
    "  border: 1px solid #38385a; border-radius: 8px;"
    "  padding: 10px 20px; font-size: 14px;"
    "  transition: background 120ms; }"
    "button:hover { background: #2c2c48; color: #ffffff; }"
    "button:active { background: #18182c; }"
    ".btn-primary {"
    "  background: #0d5f48; color: #ffffff;"
    "  border-color: #45e0b8; font-weight: bold; font-size: 14px; }"
    ".btn-primary:hover { background: #126655; }"
    ".btn-danger { background: #4a1818; color: #ff8080;"
    "  border-color: #6a2828; font-size: 14px; }"
    ".btn-danger:hover { background: #5a2020; }"

    /* === Slider === */
    "scale { color: #ccc; }"
    "scale trough { background: #232338; border-radius: 4px; min-height: 6px; }"
    "scale highlight { background: #45e0b8; border-radius: 4px; }"
    "scale slider {"
    "  background: #ffffff; border-radius: 50%;"
    "  min-width: 20px; min-height: 20px;"
    "  border: 2px solid #45e0b8; }"

    /* === Slider section === */
    ".slider-section {"
    "  background: #1c1c28; border-radius: 10px;"
    "  padding: 12px 16px; border: 1px solid #2c2c42; }"
    ".slider-label { font-size: 13px; color: #9090aa; margin-bottom: 8px; }"

    /* ============================================================
     *  SCHEDULES TAB — explicit dark background + high-contrast text
     *  The root issue was white GTK background bleeding through.
     * ============================================================ */
    /* Force the tab itself to be dark */
    "notebook > stack { background: #16161f; }"

    /* Schedule header bar */
    ".sched-header {"
    "  background: #0e0e16; border-bottom: 2px solid #28283a;"
    "  padding: 14px 20px; }"
    ".sched-count { font-size: 17px; font-weight: bold; color: #e0e0f0; }"

    /* List box and rows — ALL dark */
    "listbox { background: #16161f; }"
    "listboxrow {"
    "  background: #1c1c28; border-radius: 10px;"
    "  margin: 4px 0; border: 1px solid #2c2c42; }"
    "listboxrow:selected { background: #1a2a3e; border-color: #45e0b8; }"
    "listboxrow:hover:not(:selected) { background: #202038; }"

    /* Schedule row text — explicit bright colours */
    ".sched-name { font-size: 15px; font-weight: bold; color: #e8e8f8; }"
    ".sched-meta { font-size: 13px; color: #9898b8; font-family: monospace; }"
    ".dot-on  { font-size: 18px; color: #45e0b8; }"
    ".dot-off { font-size: 18px; color: #444466; }"
    ".temp-badge {"
    "  background: #182e28; color: #45e0b8;"
    "  border-radius: 14px; padding: 4px 14px;"
    "  font-size: 14px; font-weight: bold;"
    "  border: 1px solid #2a4e42; }"
    ".empty-hint { color: #666688; font-size: 15px; font-style: italic; }"

    /* Column header labels */
    ".col-header { font-size: 12px; color: #6666aa; font-weight: bold; }"

    /* === Section headings === */
    ".section-title { font-size: 12px; color: #5555aa; margin: 10px 0 6px; }"

    /* === Status bar === */
    ".statusbar {"
    "  background: #0a0a12; color: #6666aa; font-size: 13px;"
    "  padding: 6px 16px; border-top: 1px solid #222236; }"

    /* === Chart panel === */
    ".chart-frame {"
    "  background: #10101a; border-radius: 10px;"
    "  border: 1px solid #28283a; }"
    ".chart-title { font-size: 13px; color: #6666aa; margin: 8px 0 0 8px; }"

    /* === Separator === */
    "separator { background: #28283a; min-width: 1px; }"

    /* === Scrolled window — dark === */
    "scrolledwindow { background: #16161f; }"
    "scrolledwindow viewport { background: #16161f; }";

    gtk_css_provider_load_from_data(css, style, -1, NULL);
    gtk_style_context_add_provider_for_screen(
        gdk_screen_get_default(),
        GTK_STYLE_PROVIDER(css),
        GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    g_object_unref(css);
}

/* ============================================================
 *  HELPERS
 * ============================================================ */
static GtkWidget *make_stat_card(const char *title, GtkWidget **val_label,
                                  const char *val_class)
{
    GtkWidget *card = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
    GtkStyleContext *ctx = gtk_widget_get_style_context(card);
    gtk_style_context_add_class(ctx, "stat-card");

    GtkWidget *title_lbl = gtk_label_new(title);
    gtk_label_set_xalign(GTK_LABEL(title_lbl), 0);
    ctx = gtk_widget_get_style_context(title_lbl);
    gtk_style_context_add_class(ctx, "stat-card-label");
    gtk_box_pack_start(GTK_BOX(card), title_lbl, FALSE, FALSE, 0);

    *val_label = gtk_label_new("--");
    gtk_label_set_xalign(GTK_LABEL(*val_label), 0);
    ctx = gtk_widget_get_style_context(*val_label);
    gtk_style_context_add_class(ctx, val_class);
    gtk_box_pack_start(GTK_BOX(card), *val_label, FALSE, FALSE, 0);

    return card;
}

static GtkWidget *make_btn(const char *label, GCallback cb,
                             const char *css_class)
{
    GtkWidget *b = gtk_button_new_with_label(label);
    if (css_class) {
        GtkStyleContext *ctx = gtk_widget_get_style_context(b);
        gtk_style_context_add_class(ctx, css_class);
    }
    if (cb) g_signal_connect(b, "clicked", cb, NULL);
    return b;
}

/* ============================================================
 *  UI BUILDER  v3.2
 *
 *  Changes vs v3.1:
 *    • Sidebar: 300px wide (was 220)
 *    • Chart: 180px tall (was 260) → more room for sidebar content
 *    • Button grid: 2 rows of 4 with generous spacing
 *    • Tab 3: dark header bar, dark scrolled window, buttons at TOP
 * ============================================================ */
void build_ui(GtkApplication *app)
{
    apply_css();

    window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(window), "AI Thermostat  v" VERSION);
    gtk_window_set_default_size(GTK_WINDOW(window), 1120, 720);
    g_signal_connect(window, "destroy", G_CALLBACK(on_window_destroy), NULL);

    GtkWidget *root = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_container_add(GTK_CONTAINER(window), root);

    GtkWidget *nb = gtk_notebook_new();
    gtk_notebook_set_tab_pos(GTK_NOTEBOOK(nb), GTK_POS_TOP);
    gtk_box_pack_start(GTK_BOX(root), nb, TRUE, TRUE, 0);

    /* ============================================================
     *  TAB 1 — DASHBOARD
     * ============================================================ */
    GtkWidget *tab1 = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);

    /* --- SIDEBAR  (300px) --- */
    GtkWidget *sidebar = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
    gtk_widget_set_size_request(sidebar, 300, -1);
    gtk_container_set_border_width(GTK_CONTAINER(sidebar), 14);
    GtkStyleContext *sctx = gtk_widget_get_style_context(sidebar);
    gtk_style_context_add_class(sctx, "sidebar");
    gtk_box_pack_start(GTK_BOX(tab1), sidebar, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(tab1),
        gtk_separator_new(GTK_ORIENTATION_VERTICAL), FALSE, FALSE, 0);

    gtk_box_pack_start(GTK_BOX(sidebar),
        make_stat_card("Simulation time", &lbl_time, "stat-card-val"),
        FALSE, FALSE, 0);

    gtk_box_pack_start(GTK_BOX(sidebar),
        make_stat_card("Indoor temperature", &lbl_indoor, "big-temp"),
        FALSE, FALSE, 0);

    GtkWidget *sp_out = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    gtk_box_pack_start(GTK_BOX(sp_out),
        make_stat_card("Setpoint", &lbl_setpoint, "stat-card-val"),
        TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(sp_out),
        make_stat_card("Outside", &lbl_outside, "stat-card-val"),
        TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(sidebar), sp_out, FALSE, FALSE, 0);

    /* HVAC card */
    GtkWidget *hvac_card = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    sctx = gtk_widget_get_style_context(hvac_card);
    gtk_style_context_add_class(sctx, "stat-card");
    GtkWidget *hvac_title = gtk_label_new("HVAC status");
    gtk_label_set_xalign(GTK_LABEL(hvac_title), 0);
    sctx = gtk_widget_get_style_context(hvac_title);
    gtk_style_context_add_class(sctx, "stat-card-label");
    gtk_box_pack_start(GTK_BOX(hvac_card), hvac_title, FALSE, FALSE, 0);
    lbl_heating = gtk_label_new("—");
    gtk_label_set_xalign(GTK_LABEL(lbl_heating), 0);
    sctx = gtk_widget_get_style_context(lbl_heating);
    gtk_style_context_add_class(sctx, "hvac-label");
    gtk_box_pack_start(GTK_BOX(hvac_card), lbl_heating, FALSE, FALSE, 0);
    lbl_cooling = gtk_label_new("—");
    gtk_label_set_xalign(GTK_LABEL(lbl_cooling), 0);
    sctx = gtk_widget_get_style_context(lbl_cooling);
    gtk_style_context_add_class(sctx, "hvac-label");
    gtk_box_pack_start(GTK_BOX(hvac_card), lbl_cooling, FALSE, FALSE, 0);
    lbl_occup = gtk_label_new("—");
    gtk_label_set_xalign(GTK_LABEL(lbl_occup), 0);
    sctx = gtk_widget_get_style_context(lbl_occup);
    gtk_style_context_add_class(sctx, "hvac-label");
    gtk_box_pack_start(GTK_BOX(hvac_card), lbl_occup, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(sidebar), hvac_card, FALSE, FALSE, 0);

    gtk_box_pack_start(GTK_BOX(sidebar),
        make_stat_card("Energy", &lbl_energy, "stat-card-val"),
        FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(sidebar),
        make_stat_card("AI interactions", &lbl_interactions, "stat-card-val"),
        FALSE, FALSE, 0);

    /* --- MAIN CONTENT --- */
    GtkWidget *main_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 14);
    gtk_container_set_border_width(GTK_CONTAINER(main_box), 16);
    gtk_box_pack_start(GTK_BOX(tab1), main_box, TRUE, TRUE, 0);

    /* Chart — reduced height: 180px */
    GtkWidget *chart_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    sctx = gtk_widget_get_style_context(chart_box);
    gtk_style_context_add_class(sctx, "chart-frame");
    GtkWidget *chart_lbl = gtk_label_new(
        "Temperature  —  history (solid)  +  forecast (dashed dots)");
    gtk_label_set_xalign(GTK_LABEL(chart_lbl), 0);
    sctx = gtk_widget_get_style_context(chart_lbl);
    gtk_style_context_add_class(sctx, "chart-title");
    gtk_box_pack_start(GTK_BOX(chart_box), chart_lbl, FALSE, FALSE, 4);
    drawing_chart = gtk_drawing_area_new();
    gtk_widget_set_size_request(drawing_chart, -1, 180);
    g_signal_connect(drawing_chart, "draw", G_CALLBACK(on_draw_chart), NULL);
    gtk_box_pack_start(GTK_BOX(chart_box), drawing_chart, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(main_box), chart_box, FALSE, FALSE, 0);

    /* Slider section */
    GtkWidget *slider_card = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
    sctx = gtk_widget_get_style_context(slider_card);
    gtk_style_context_add_class(sctx, "slider-section");
    GtkWidget *slider_title = gtk_label_new("Manual setpoint control");
    gtk_label_set_xalign(GTK_LABEL(slider_title), 0);
    sctx = gtk_widget_get_style_context(slider_title);
    gtk_style_context_add_class(sctx, "slider-label");
    gtk_box_pack_start(GTK_BOX(slider_card), slider_title, FALSE, FALSE, 0);

    GtkWidget *slider_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    GtkWidget *sl_lo = gtk_label_new("10°");
    sctx = gtk_widget_get_style_context(sl_lo);
    gtk_style_context_add_class(sctx, "stat-card-label");
    scale_setpoint = gtk_scale_new_with_range(
        GTK_ORIENTATION_HORIZONTAL, TEMP_MIN, TEMP_MAX, 0.5);
    gtk_scale_set_value_pos(GTK_SCALE(scale_setpoint), GTK_POS_RIGHT);
    gtk_scale_set_digits(GTK_SCALE(scale_setpoint), 1);
    gtk_range_set_value(GTK_RANGE(scale_setpoint), T.current_setpoint);
    GtkWidget *sl_hi = gtk_label_new("35°");
    sctx = gtk_widget_get_style_context(sl_hi);
    gtk_style_context_add_class(sctx, "stat-card-label");
    gtk_box_pack_start(GTK_BOX(slider_row), sl_lo,         FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(slider_row), scale_setpoint, TRUE,  TRUE,  0);
    gtk_box_pack_start(GTK_BOX(slider_row), sl_hi,         FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(slider_card), slider_row, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(main_box), slider_card, FALSE, FALSE, 0);

    /* Button grid — 2 rows × 4 cols with generous spacing */
    GtkWidget *btn_grid = gtk_grid_new();
    gtk_grid_set_column_spacing(GTK_GRID(btn_grid), 12);
    gtk_grid_set_row_spacing(GTK_GRID(btn_grid), 10);
    gtk_grid_set_column_homogeneous(GTK_GRID(btn_grid), TRUE);

    gtk_grid_attach(GTK_GRID(btn_grid),
        make_btn("✔  Apply setpoint",   G_CALLBACK(on_set_manual),      "btn-primary"), 0,0,1,1);
    gtk_grid_attach(GTK_GRID(btn_grid),
        make_btn("🤖  AI optimise",     G_CALLBACK(on_ai_setpoint),     "btn-primary"), 1,0,1,1);
    gtk_grid_attach(GTK_GRID(btn_grid),
        make_btn("👤  Toggle occupancy",G_CALLBACK(on_toggle_occupancy), NULL),          2,0,1,1);
    gtk_grid_attach(GTK_GRID(btn_grid),
        make_btn("💾  Save",            G_CALLBACK(on_save),             NULL),          3,0,1,1);
    gtk_grid_attach(GTK_GRID(btn_grid),
        make_btn("⏩  + 1 hour",        G_CALLBACK(on_advance_1h),       NULL),          0,1,1,1);
    gtk_grid_attach(GTK_GRID(btn_grid),
        make_btn("⏩  + 6 hours",       G_CALLBACK(on_advance_6h),       NULL),          1,1,1,1);
    gtk_grid_attach(GTK_GRID(btn_grid),
        make_btn("⏩  + 24 hours",      G_CALLBACK(on_advance_24h),      NULL),          2,1,1,1);

    gtk_box_pack_start(GTK_BOX(main_box), btn_grid, FALSE, FALSE, 0);

    gtk_notebook_append_page(GTK_NOTEBOOK(nb), tab1,
                             gtk_label_new("  Dashboard  "));

    /* ============================================================
     *  TAB 2 — COMFORT PROFILE
     * ============================================================ */
    GtkWidget *tab2 = gtk_box_new(GTK_ORIENTATION_VERTICAL, 12);
    gtk_container_set_border_width(GTK_CONTAINER(tab2), 16);

    GtkWidget *cp_hdr = gtk_label_new(
        "Learned 24-hour comfort profile — "
        "bar height = preferred temp,  "
        "color = temperature (blue = cold, orange = warm),  "
        "dots = confidence");
    gtk_label_set_xalign(GTK_LABEL(cp_hdr), 0);
    gtk_label_set_line_wrap(GTK_LABEL(cp_hdr), TRUE);
    sctx = gtk_widget_get_style_context(cp_hdr);
    gtk_style_context_add_class(sctx, "stat-card-label");
    gtk_box_pack_start(GTK_BOX(tab2), cp_hdr, FALSE, FALSE, 0);

    GtkWidget *comfort_frame = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    sctx = gtk_widget_get_style_context(comfort_frame);
    gtk_style_context_add_class(sctx, "chart-frame");
    drawing_comfort = gtk_drawing_area_new();
    gtk_widget_set_size_request(drawing_comfort, -1, 420);
    g_signal_connect(drawing_comfort, "draw", G_CALLBACK(on_draw_comfort), NULL);
    gtk_box_pack_start(GTK_BOX(comfort_frame), drawing_comfort, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(tab2), comfort_frame, TRUE, TRUE, 0);

    GtkWidget *hint = gtk_label_new(
        "White outline = current hour.  "
        "Set temperatures manually on the Dashboard to train the model.  "
        "Grey bars = no data yet for that hour.");
    gtk_label_set_line_wrap(GTK_LABEL(hint), TRUE);
    gtk_label_set_xalign(GTK_LABEL(hint), 0);
    sctx = gtk_widget_get_style_context(hint);
    gtk_style_context_add_class(sctx, "stat-card-label");
    gtk_box_pack_start(GTK_BOX(tab2), hint, FALSE, FALSE, 0);

    gtk_notebook_append_page(GTK_NOTEBOOK(nb), tab2,
                             gtk_label_new("  Comfort Profile  "));

    /* ============================================================
     *  TAB 3 — SCHEDULES  (dark throughout, buttons at top)
     * ============================================================ */
    GtkWidget *tab3 = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);

    /* Header bar — dark */
    GtkWidget *sched_hdr = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
    sctx = gtk_widget_get_style_context(sched_hdr);
    gtk_style_context_add_class(sctx, "sched-header");

    lbl_sched_count = gtk_label_new("Schedules");
    sctx = gtk_widget_get_style_context(lbl_sched_count);
    gtk_style_context_add_class(sctx, "sched-count");
    gtk_box_pack_start(GTK_BOX(sched_hdr), lbl_sched_count, FALSE, FALSE, 0);

    /* Spacer to push buttons right */
    gtk_box_pack_start(GTK_BOX(sched_hdr), gtk_label_new(""), TRUE, TRUE, 0);

    GtkWidget *btn_add = make_btn("+ Add",    G_CALLBACK(on_add_schedule),    "btn-primary");
    GtkWidget *btn_edt = make_btn("✎ Edit",   G_CALLBACK(on_edit_schedule),   NULL);
    GtkWidget *btn_del = make_btn("✕ Delete", G_CALLBACK(on_delete_schedule), "btn-danger");
    gtk_box_pack_start(GTK_BOX(sched_hdr), btn_add, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(sched_hdr), btn_edt, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(sched_hdr), btn_del, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(tab3), sched_hdr, FALSE, FALSE, 0);

    /* Column header row */
    GtkWidget *col_hdr = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 14);
    gtk_container_set_border_width(GTK_CONTAINER(col_hdr), 10);
    gtk_widget_set_margin_start(col_hdr, 16);
    const char *col_names[] = { "  ", "Name", "Day", "Time range", NULL };
    int col_widths[]        = {  22,   170,    110,   120 };
    int ci;
    for (ci = 0; col_names[ci]; ci++) {
        GtkWidget *cl = gtk_label_new(col_names[ci]);
        gtk_label_set_xalign(GTK_LABEL(cl), 0);
        gtk_widget_set_size_request(cl, col_widths[ci], -1);
        sctx = gtk_widget_get_style_context(cl);
        gtk_style_context_add_class(sctx, "col-header");
        gtk_box_pack_start(GTK_BOX(col_hdr), cl, FALSE, FALSE, 0);
    }
    GtkWidget *cl_temp = gtk_label_new("Temp");
    gtk_label_set_xalign(GTK_LABEL(cl_temp), 1);
    sctx = gtk_widget_get_style_context(cl_temp);
    gtk_style_context_add_class(sctx, "col-header");
    gtk_box_pack_end(GTK_BOX(col_hdr), cl_temp, FALSE, FALSE, 18);
    gtk_box_pack_start(GTK_BOX(tab3), col_hdr, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(tab3),
        gtk_separator_new(GTK_ORIENTATION_HORIZONTAL), FALSE, FALSE, 0);

    /* Scrolled list */
    GtkWidget *scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll),
                                   GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_container_set_border_width(GTK_CONTAINER(scroll), 12);

    schedule_list = gtk_list_box_new();
    gtk_list_box_set_selection_mode(GTK_LIST_BOX(schedule_list),
                                    GTK_SELECTION_SINGLE);
    gtk_container_add(GTK_CONTAINER(scroll), schedule_list);
    gtk_box_pack_start(GTK_BOX(tab3), scroll, TRUE, TRUE, 0);

    gtk_notebook_append_page(GTK_NOTEBOOK(nb), tab3,
                             gtk_label_new("  Schedules  "));

    /* ============================================================
     *  STATUS BAR
     * ============================================================ */
    lbl_status_bar = gtk_label_new(
        load_state() ? "Session restored from thermostat.dat"
                     : "First run — data saves automatically on exit");
    gtk_label_set_xalign(GTK_LABEL(lbl_status_bar), 0);
    sctx = gtk_widget_get_style_context(lbl_status_bar);
    gtk_style_context_add_class(sctx, "statusbar");
    gtk_box_pack_end(GTK_BOX(root), lbl_status_bar, FALSE, FALSE, 0);
    gtk_box_pack_end(GTK_BOX(root),
        gtk_separator_new(GTK_ORIENTATION_HORIZONTAL), FALSE, FALSE, 0);

    refresh_status();
    refresh_schedules_list();

    g_timeout_add(1000, on_tick, NULL);
    gtk_widget_show_all(window);
}
// CEOF
// echo "Written: $(wc -l < /home/claude/thermostat_gtk.c) lines"
 




