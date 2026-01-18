import './Sidebar.scss';

interface SidebarProps {
    children: React.ReactNode;
}

export default function Sidebar({ children }: SidebarProps) {
    return (
        <aside className="sidebar">
            {children}
        </aside>
    );
}
